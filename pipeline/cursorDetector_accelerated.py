import cv2
import pandas as pd
import os
import time
import threading
import queue
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import torch

# --- Acceleration Modules ---
HAS_EXTENSION = False
try:
    import lol_accelerator
    HAS_EXTENSION = True
    print("✅ C++ Extension 'lol_accelerator' loaded successfully.")
except ImportError:
    print("⚠️ 'lol_accelerator' extension not found. Falling back to Numba/Python implementation.")

# Check for Numba
HAS_NUMBA = False
try:
    from numba import cuda
    if cuda.is_available():
        HAS_NUMBA = True
        print("✅ Numba CUDA support detected.")
    else:
        print("⚠️ Numba installed but CUDA device not available.")
except ImportError:
    print("⚠️ Numba not installed.")

# --- Custom CUDA Kernel (Numba Version) ---
if HAS_NUMBA and not HAS_EXTENSION:
    @cuda.jit
    def highlight_cursor_kernel_numba(input_array, output_array):
        x, y = cuda.grid(2)
        height, width, channels = input_array.shape
        
        if x < width and y < height:
            # BGR assumed
            b = input_array[y, x, 0]
            g = input_array[y, x, 1]
            r = input_array[y, x, 2]
            
            luminance = 0.114 * b + 0.587 * g + 0.299 * r
            
            if luminance > 200:
                output_array[y, x, 0] = min(255, int(b * 1.2))
                output_array[y, x, 1] = min(255, int(g * 1.2))
                output_array[y, x, 2] = min(255, int(r * 1.2))
            else:
                output_array[y, x, 0] = int(b * 0.8)
                output_array[y, x, 1] = int(g * 0.8)
                output_array[y, x, 2] = int(r * 0.8)

def preprocess_frame_cuda(frame):
    """
    Apply custom CUDA preprocessing.
    """
    if HAS_EXTENSION:
        # Convert to torch tensor on GPU
        tensor_img = torch.from_numpy(frame).to('cuda')
        # processed = lol_accelerator.preprocess_cuda(tensor_img) 
        # For now, since extension needs compilation, we skip actual call if mocking
        return frame # Placeholder until extension is compiled
        
    elif HAS_NUMBA:
        # Numba implementation
        # Upload to GPU
        d_input = cuda.to_device(frame)
        d_output = cuda.device_array_like(frame)
        
        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(frame.shape[1] / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(frame.shape[0] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        highlight_cursor_kernel_numba[blockspergrid, threadsperblock](d_input, d_output)
        
        # Download result (or keep on GPU if YOLO accepts it, but Ultralytics usually takes numpy or file or torch)
        # Here we return numpy for compatibility
        return d_output.copy_to_host()
        
    return frame # Fallback: No op

# --- Async Video Loader (Python Threaded Version) ---
class ThreadedVideoLoader:
    def __init__(self, path, queue_size=30):
        self.cap = cv2.VideoCapture(path)
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def update(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.q.put(frame)
            else:
                time.sleep(0.005)
        self.cap.release()

    def read(self):
        if self.q.empty():
            if self.stopped:
                return False, None
            else:
                # Block slightly waiting for frame
                try:
                    frame = self.q.get(timeout=1.0)
                    return True, frame
                except queue.Empty:
                    return False, None
        return True, self.q.get()

    def stop(self):
        self.stopped = True
        self.thread.join()

# --- Main Processing Logic ---

def process_videos_accelerated(model_path, record_video, record_csv, selected_videos):
    print(f"Loading YOLO model: {model_path}")
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path) 
    
    for video in selected_videos:
        print(f"Processing {os.path.basename(video)}...")
        
        # Use Async Loader
        if HAS_EXTENSION:
             # In real scenario: loader = lol_accelerator.AsyncVideoLoader(video)
             # Fallback to python threaded for now as extension compilation is complex
             loader = ThreadedVideoLoader(video)
        else:
             loader = ThreadedVideoLoader(video)
             
        cursor_data = []
        
        out = None
        if record_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = f'processed_videos/{os.path.basename(video)[:-4]}_accelerated.mp4'
            out = cv2.VideoWriter(out_path, fourcc, loader.fps, (loader.width, loader.height))

        frame_idx = 0
        start_time = time.time()
        
        while True:
            ret, frame = loader.read()
            if not ret or frame is None:
                break

            # 1. Custom CUDA Preprocessing
            processed_frame = preprocess_frame_cuda(frame)

            # 2. YOLO Inference
            # passing half=True for FP16 speedup if on CUDA
            results = model(processed_frame, device=device, half=(device=='cuda'), verbose=False, stream=True)
            
            # Ultralytics stream=True returns a generator, need to iterate
            for result in results:
                boxes = result.boxes
                if len(boxes) > 0:
                    # Find highest confidence box
                    # boxes.conf is a tensor
                    confs = boxes.conf.cpu().numpy()
                    best_idx = np.argmax(confs)
                    
                    if confs[best_idx] > 0.5:
                        xyxy = boxes.xyxy[best_idx].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        conf = confs[best_idx]
                        
                        if record_csv:
                            cursor_data.append([frame_idx / loader.fps, x1, y1])
                        
                        if record_video:
                            # Draw on original frame (not processed one usually, to keep colors natural)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f'{conf:.2f}', (int(x1), int(y1)-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if record_video:
                out.write(frame)
                
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames...", end='\r')

        total_time = time.time() - start_time
        print(f"\nFinished {video}. Average FPS: {frame_idx / total_time:.2f}")

        loader.stop()
        if out:
            out.release()

        if record_csv:
            df = pd.DataFrame(cursor_data, columns=['Time', 'X', 'Y'])
            df.to_csv(f'mouse_positions/{os.path.basename(video)[:-4]}.csv', index=False)

def select_files():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title='Select Video Files for Accelerated Processing')
    return list(file_paths)

if __name__ == "__main__":
    os.makedirs('processed_videos', exist_ok=True)
    os.makedirs('mouse_positions', exist_ok=True)

    print("=== LoLTrackGuard High-Performance Mode ===")
    print("1. Select video files via dialog")
    
    videos = select_files()
    if not videos:
        print("No videos selected.")
    else:
        rec_video = input("Record annotated video? (y/n): ").lower() == 'y'
        rec_csv = input("Record mouse CSV? (y/n): ").lower() == 'y'
        
        # Assuming default model path
        model_path = 'utils/detector.pt'
        if not os.path.exists(model_path):
             # Fallback to creating or asking, but let's try to find one
             model_path = 'utils/cursorDetector.pt' # Try alternate name if exists
             if not os.path.exists(model_path):
                 model_path = 'yolov8n.pt' # Last resort
        
        process_videos_accelerated(model_path, rec_video, rec_csv, videos)

