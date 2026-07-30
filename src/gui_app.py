import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import sys

import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np


# ---- Настройки ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 224
BATCH_SIZE = 32
DEFAULT_THRESHOLD = 0.5


# Определение базовой папки (учёт запуска как .py и как .exe)
if hasattr(sys, '_MEIPASS'):
    # Запуск из exe, PyInstaller распаковал временную папку
    BASE_DIR = sys._MEIPASS
else:
    # Обычный запуск .py
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Корень проекта = родительская папка относительно src/
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_efficientnet_b0_aug2_newdataset.pth"
)


def load_model(model_path):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()
    return model


def build_transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def open_folder(path):
    """
    Открыть папку в проводнике (Windows).
    """
    if not os.path.isdir(path):
        messagebox.showwarning("Предупреждение", f"Папка не найдена:\n{path}")
        return
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform.startswith("darwin"):
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def classify_unlabeled_folder(model, folder_path, threshold=DEFAULT_THRESHOLD):
    """
    Неразмеченные фото: все изображения в одной папке.
    Создаём внутри папки две подпапки и копируем туда файлы по предсказанию.
    """
    transform = build_transform()

    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    file_list = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in exts
    ]

    if not file_list:
        messagebox.showwarning("Предупреждение", "В выбранной папке нет изображений.")
        return None

    civilian_dir = os.path.join(folder_path, "civilian_pred")
    military_dir = os.path.join(folder_path, "military_pred")
    os.makedirs(civilian_dir, exist_ok=True)
    os.makedirs(military_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for path in file_list:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(DEVICE)

            output = model(img_t).squeeze(1)
            prob = torch.sigmoid(output).item()
            pred = 1 if prob >= threshold else 0

            if pred == 0:
                dst_dir = civilian_dir
            else:
                dst_dir = military_dir

            shutil.copy(path, os.path.join(dst_dir, os.path.basename(path)))

    messagebox.showinfo(
        "Готово",
        f"Классификация завершена.\n"
        f"Файлы разложены по папкам:\n"
        f"- {civilian_dir}\n- {military_dir}"
    )
    return folder_path  # как корневой путь с результатами


def classify_labeled_folder(model, root_dir, threshold=DEFAULT_THRESHOLD):
    """
    Размеченные фото: структура root_dir/civilian и root_dir/military.
    Считаем метрики, показываем их и копируем ошибочные примеры
    в отдельные папки внутри root_dir.
    """
    transform = build_transform()

    civilian_path = os.path.join(root_dir, "civilian")
    military_path = os.path.join(root_dir, "military")
    if not (os.path.isdir(civilian_path) and os.path.isdir(military_path)):
        messagebox.showerror(
            "Ошибка",
            "В выбранной папке должны быть подпапки 'civilian' и 'military'."
        )
        return None

    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_probs = []
    all_preds = []
    all_labels = []
    all_paths = []

    # список исходных путей
    for path, _ in dataset.samples:
        all_paths.append(path)

    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(targets.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    report = classification_report(
        y_true, y_pred,
        target_names=["civilian", "military"],
        digits=4
    )
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    # --- копирование ошибочных примеров ---
    errors_dir = os.path.join(root_dir, "errors")
    fp_dir = os.path.join(errors_dir, "false_positive")   # гражданская, предсказано военная
    fn_dir = os.path.join(errors_dir, "false_negative")   # военная, предсказано гражданская
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    for path, true_lbl, pred_lbl in zip(all_paths, all_labels, all_preds):
        # true_lbl: 0 - civilian, 1 - military
        # pred_lbl: 0 - civilian, 1 - military
        if true_lbl == 0 and pred_lbl == 1:
            # гражданская → военная (ложноположительное срабатывание)
            shutil.copy(path, os.path.join(fp_dir, os.path.basename(path)))
        elif true_lbl == 1 and pred_lbl == 0:
            # военная → гражданская (пропуск военного объекта)
            shutil.copy(path, os.path.join(fn_dir, os.path.basename(path)))

    # окно с отчётом
    result_window = tk.Toplevel()
    result_window.title("Результаты классификации (размеченные данные)")

    text = tk.Text(result_window, width=80, height=25)
    text.pack(fill=tk.BOTH, expand=True)

    text.insert(tk.END, f"Порог принятия решения: {threshold}\n\n")
    text.insert(tk.END, "Отчёт по метрикам:\n")
    text.insert(tk.END, report + "\n\n")
    text.insert(tk.END, "Матрица ошибок (формат [[TN FP]\n [FN TP]]):\n")
    text.insert(tk.END, f"{cm}\n\n")
    text.insert(tk.END, f"TN={TN}, FP={FP}, FN={FN}, TP={TP}\n\n")
    text.insert(
        tk.END,
        f"Ошибочно классифицированные изображения сохранены в папке:\n{errors_dir}\n"
    )

    text.config(state=tk.DISABLED)

    messagebox.showinfo(
        "Готово",
        f"Классификация завершена.\nОшибки сохранены в папке:\n{errors_dir}"
    )

    return errors_dir  # путь к результатам (ошибки)


class App:
    def __init__(self, master):
        self.master = master
        master.title("Классификация гражданской и военной техники")
        master.geometry("650x350")

        self.model = None

        # выбранный режим и папка/путь с результатами
        self.mode = tk.StringVar(value="unlabeled")  # 'unlabeled' или 'labeled'
        self.selected_folder = None
        self.results_folder = None

        # Заголовок
        lbl_title = tk.Label(
            master,
            text="Классификация изображений гражданской и военной техники",
            font=("Arial", 12, "bold")
        )
        lbl_title.pack(pady=10)

        # Выбор режима
        frame_mode = tk.Frame(master)
        frame_mode.pack(pady=5)

        tk.Label(frame_mode, text="Режим работы:").pack(side=tk.LEFT)

        tk.Radiobutton(
            frame_mode,
            text="Неразмеченные фото (одна папка)",
            variable=self.mode,
            value="unlabeled"
        ).pack(side=tk.LEFT, padx=5)

        tk.Radiobutton(
            frame_mode,
            text="Размеченные (civilian / military)",
            variable=self.mode,
            value="labeled"
        ).pack(side=tk.LEFT, padx=5)

        # Порог
        frame_threshold = tk.Frame(master)
        frame_threshold.pack(pady=5)

        tk.Label(frame_threshold, text="Порог принятия решения (0..1):").pack(side=tk.LEFT)
        self.threshold_var = tk.StringVar(value=str(DEFAULT_THRESHOLD))
        tk.Entry(frame_threshold, textvariable=self.threshold_var, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(frame_threshold, text="(рекомендуемый 0.5)", fg="gray").pack(side=tk.LEFT, padx=5)

        # Выбор папки
        frame_folder = tk.Frame(master)
        frame_folder.pack(pady=10)

        btn_choose = tk.Button(
            frame_folder,
            text="Выбрать папку с изображениями",
            command=self.choose_folder
        )
        btn_choose.pack(side=tk.LEFT)

        self.lbl_folder = tk.Label(
            frame_folder,
            text="Папка не выбрана",
            fg="gray"
        )
        self.lbl_folder.pack(side=tk.LEFT, padx=10)

        # Кнопки управления
        frame_buttons = tk.Frame(master)
        frame_buttons.pack(pady=15)

        btn_start = tk.Button(
            frame_buttons,
            text="Начать обработку",
            width=20,
            command=self.start_processing
        )
        btn_start.pack(side=tk.LEFT, padx=10)

        btn_open_results = tk.Button(
            frame_buttons,
            text="Открыть папку с результатами",
            width=25,
            command=self.open_results_folder
        )
        btn_open_results.pack(side=tk.LEFT, padx=10)

        # Статус
        self.lbl_status = tk.Label(master, text=f"Устройство: {DEVICE}", fg="gray")
        self.lbl_status.pack(side=tk.BOTTOM, pady=5)

        # Загружаем модель сразу
        self.load_model_once()

    def load_model_once(self):
        try:
            self.model = load_model(MODEL_PATH)
            self.lbl_status.config(text=f"Модель загружена. Устройство: {DEVICE}")
        except Exception as e:
            messagebox.showerror("Ошибка загрузки модели", str(e))
            self.lbl_status.config(text="Ошибка загрузки модели")

    def get_threshold(self):
        try:
            t = float(self.threshold_var.get().replace(",", "."))
            if not (0.0 <= t <= 1.0):
                raise ValueError
            return t
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректное значение порога. Укажите число от 0 до 1.")
            return None

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку с изображениями")
        if folder:
            self.selected_folder = folder
            self.results_folder = None  # сбрасываем старые результаты
            self.lbl_folder.config(text=folder, fg="black")

    def start_processing(self):
        if self.model is None:
            messagebox.showerror("Ошибка", "Модель не загружена.")
            return

        if not self.selected_folder:
            messagebox.showwarning("Предупреждение", "Сначала выберите папку с изображениями.")
            return

        threshold = self.get_threshold()
        if threshold is None:
            return

        mode = self.mode.get()

        if mode == "unlabeled":
            # неразмеченные данные
            self.results_folder = classify_unlabeled_folder(
                self.model,
                self.selected_folder,
                threshold
            )
        else:
            # размеченные данные
            self.results_folder = classify_labeled_folder(
                self.model,
                self.selected_folder,
                threshold
            )

    def open_results_folder(self):
        if not self.results_folder:
            messagebox.showwarning(
                "Предупреждение",
                "Результаты ещё не получены. Сначала запустите обработку."
            )
            return
        open_folder(self.results_folder)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()