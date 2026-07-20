# Конвейер: сегментация товаров (этап 1)

Авто-дистилляция Grounding DINO → SAM → YOLO11-seg + ByteTrack.
Один класс: `item`. На выходе — полигоны и центроиды для контроллера робота.

## Установка

```powershell
cd C:\Users\admin\Downloads\Робозон\conveyor_seg
python -m venv .venv
.venv\Scripts\activate

# torch первым, под свою CUDA. RTX 5080 = sm_120 -> нужен cu128:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
```

Если `torch.cuda.is_available()` = False или ругается на `sm_120 is not compatible` — стоит cu121-билд, переустанови torch с индекса cu128.

## Запуск

Пути уже прописаны в `configs/project.yaml` (`paths.videos` → твоя папка с видео).

```powershell
# 1. Кадры (с фильтром смаза и дублей)
python -m src.extract_frames

# 2. Отладочный прогон разметки на 50 кадрах — ОБЯЗАТЕЛЬНО посмотреть глазами
python -m src.autolabel --limit 50 --preview
#    -> work\preview\*.jpg

# 3. Полная разметка
python -m src.autolabel

# 4. Датасет + conveyor.yaml (сплит по видео)
python -m src.build_dataset

# 5. Обучение
python -m src.train

# 6. Рантайм с трекингом
python -m src.infer_track --source "C:\Users\admin\Downloads\Робозон\conveyor_seg\video\test.mp4" --show --save-video "C:\Users\admin\Downloads\Робозон\work\out.mp4" --save-jsonl picks.jsonl
```

Или всё разом: `run_all.bat`.

## Что важно знать

**Шаг 2 — единственное место, где можно всё испортить.** Grounding DINO с промптом `item` на конвейере часто ловит саму ленту, тени и блики. Отсюда фильтры в конфиге:

| Параметр | Симптом | Куда крутить |
|---|---|---|
| `gdino.box_threshold` | много кадров пустые | вниз (0.25) |
| `gdino.box_threshold` | ловит тени/мусор | вверх (0.35–0.40) |
| `gdino.max_box_area_frac` | один бокс на весь кадр = лента | вниз (0.40) |
| `gdino.min_box_area_frac` | мелкие крапинки | вверх |
| `gdino.prompt` | не видит товар | замени на то, что реально едет: `"cardboard box. plastic bag. bottle."` |

Скрипт сам предупредит, если >30% кадров вышли пустыми.

**Веса.** В ultralytics модель называется `yolo11n-seg.pt`, а не `yolov11n-seg.pt` — команда из ТЗ упадёт при скачивании. В конфиге уже правильное имя.

**Сплит по видео, а не по кадрам.** Соседние кадры одного ролика почти идентичны. Случайный сплит по кадрам даст mAP 0.98 на валидации и провал на реальной ленте. `dataset.split_by_video: true` держит целые ролики в val.

**Аугментации.** Камера над лентой неподвижна, масштаб фиксирован. Поэтому `perspective=0`, `scale` урезан до 0.25, `mosaic` до 0.3 и отключается за 15 эпох до конца. Зато включён `flipud` (вид сверху) и сильный `hsv_v` — блики на ленте.

**Метрика, на которую смотреть — `seg.map50`, не `box.map50`.** Роботу нужны границы маски, а не бокс.

## Выход для контроллера

`--save-jsonl picks.jsonl`, по строке на событие захвата:

```json
{"event":"pick","track_id":17,"frame":412,"ts":16.48,
 "centroid_px":[640.2,352.0],"centroid_norm":[0.5002,0.5501],
 "area_px":18420.0,"conf":0.91,"polygon_norm":[[0.41,0.47],...]}
```

Событие выдаётся **один раз на `track_id`** — в момент, когда центроид пересекает линию `runtime.pick_line_y` сверху вниз. Именно за этим нужен ByteTrack: без него один товар на 10 кадрах = 10 команд роботу.

Для интеграции по сети замени запись в файл на publish в ROS2 / MQTT / gRPC — структура `rec` в `infer_track.py:~95` уже готова к сериализации.

## Файлы

| Файл | Что делает |
|---|---|
| `src/extract_frames.py` | видео → кадры, отсев смаза (Laplacian) и дублей (dHash) |
| `src/autolabel.py` | Grounding DINO → боксы → SAM → маски → `.txt` YOLO-seg, `class_id=0` |
| `src/build_dataset.py` | train/val сплит по видео + `conveyor.yaml` |
| `src/train.py` | обучение yolo11n-seg, `nc=1`, аугментации под конвейер |
| `src/infer_track.py` | **основной рантайм**: YOLO-seg + ByteTrack + JSONL |
| `src/runtime_yolo_sam.py` | рантайм строго по схеме: YOLO-детекция + MobileSAM |

## Почему основной рантайм — YOLO-seg, а не YOLO+MobileSAM

На схеме нарисован `YOLO детекция → MobileSAM → маска`. Это рабочий вариант, он в `runtime_yolo_sam.py`. Но для одноклассовой задачи на контрастном фоне он платит лишний прогон энкодера SAM на каждом кадре: ~30–60 FPS против ~200+ у `yolo11n-seg`. Если конвейер быстрый — бери `infer_track.py`. Если робот промахивается из-за грубых краёв маски (мягкая упаковка, прозрачный пластик) — переключайся на `runtime_yolo_sam.py`, там границы пиксельно точные.

## Следующий шаг

Авто-разметка — не истина в последней инстанции. После первого обучения прогони модель по кадрам, отбери 200–300 худших предсказаний, поправь их руками (CVAT / Label Studio) и дообучи. Это даёт больше, чем +50 эпох.
