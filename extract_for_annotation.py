import cv2
import os
from pathlib import Path

def extract_for_annotation(video_dir, output_dir, interval=50):
    """
    video_dir: папка с видео
    output_dir: куда сохранять кадры
    interval: каждые N кадров берем один (чтобы не было дублей)
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = [f for f in video_dir.glob("*.mp4")]
    
    for vid_path in video_files:
        print(f"Обработка: {vid_path.name}")
        cap = cv2.VideoCapture(str(vid_path))
        count = 0
        saved = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            if count % interval == 0:
                # Имя файла: имя_видео_номер_кадра.jpg
                save_path = output_dir / f"{vid_path.stem}_{count:05d}.jpg"
                cv2.imwrite(str(save_path), frame)
                saved += 1
            
            count += 1
        
        cap.release()
        print(f"  Сохранено {saved} кадров.")

if __name__ == "__main__":
    # Укажите ваши пути
    video_folder = r"C:\Users\admin\Downloads\Робозон\conveyor_seg\video"
    output_folder = r"C:\Users\admin\Downloads\Робозон\conveyor_seg\frames_to_annotate"
    
    extract_for_annotation(video_folder, output_folder, interval=3)
    print("Готово! Кадры лежат в", output_folder)