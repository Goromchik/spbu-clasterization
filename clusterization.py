import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hashlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import PCA
from matplotlib.figure import Figure


class DatasetViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Кластеризация с визуализацией")
        self.original_df = None
        self.encoded_df = None
        self.anonymized_df = None
        self.labels = None
        self.selected_features = None
        self.feature_scores = {}
        self.current_data_type = "исходные"
        self.current_features_type = "все"
        self.num_features_to_select = 5
        self.figure = None
        self.canvas = None
        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        buttons = [
            ("Загрузить датасет", self.load_dataset),
            ("Обезличить данные", self.anonymize_data),
            ("Выбрать признаки", self.optimized_add_feature_selection),
            ("Кластеризовать (все признаки)",
             lambda: self.run_isodata(use_selected_features=False, use_anonymized=False)),
            ("Кластеризовать (отобранные признаки)",
             lambda: self.run_isodata(use_selected_features=True, use_anonymized=False)),
            ("Кластеризовать (обезличенные данные)",
             lambda: self.run_isodata(use_selected_features=False, use_anonymized=True)),
            ("Оценить качество", self.evaluate_clusters),
            ("Показать графики", self.show_cluster_plots)
        ]

        for text, command in buttons:
            tk.Button(control_frame, text=text, command=command).pack(pady=5, fill=tk.X)

        params = [
            ("Желаемое число кластеров:", "5"),
            ("Мин. точек в кластере:", "10"),
            ("Макс. дисперсия:", "0.5"),
            ("Мин. расстояние между кластерами:", "3.0"),
            ("Макс. итераций:", "100")
        ]

        self.entries = {}
        for label, default in params:
            frame = tk.Frame(control_frame)
            frame.pack(fill=tk.X, pady=2)
            tk.Label(frame, text=label, anchor="w").pack(fill=tk.X)
            entry = tk.Entry(frame, width=10)
            entry.insert(0, default)
            entry.pack(fill=tk.X)
            self.entries[label] = entry

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        info_frame = tk.Frame(right_frame)
        info_frame.pack(fill=tk.X)

        self.cluster_label = tk.Label(info_frame, text="Кластеры: 0", font=('Arial', 10, 'bold'),
                                    bg='lightgreen', relief=tk.RAISED, padx=5, pady=2)
        self.cluster_label.pack(side=tk.LEFT, padx=5)

        self.quality_label = tk.Label(info_frame, text="Качество: -", font=('Arial', 10, 'bold'),
                                    bg='lightblue', relief=tk.RAISED, padx=5, pady=2)
        self.quality_label.pack(side=tk.LEFT, padx=5)

        self.features_label = tk.Label(info_frame, text="Признаки: все", font=('Arial', 10, 'bold'),
                                     bg='lightyellow', relief=tk.RAISED, padx=5, pady=2)
        self.features_label.pack(side=tk.LEFT, padx=5)

        self.data_label = tk.Label(info_frame, text="Данные: исходные", font=('Arial', 10, 'bold'),
                                 bg='lightpink', relief=tk.RAISED, padx=5, pady=2)
        self.data_label.pack(side=tk.LEFT, padx=5)

        display_frame = tk.Frame(right_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)

        text_frame = tk.Frame(display_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scroll_y = tk.Scrollbar(text_frame)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        scroll_x = tk.Scrollbar(text_frame, orient=tk.HORIZONTAL)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.text_area = tk.Text(text_frame, wrap=tk.NONE, font=('Courier', 10),
                               xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
        self.text_area.pack(fill=tk.BOTH, expand=True)

        scroll_x.config(command=self.text_area.xview)
        scroll_y.config(command=self.text_area.yview)

        self.plot_frame = tk.Frame(display_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

    def show_cluster_plots(self):
        if self.labels is None:
            messagebox.showwarning("Ошибка", "Сначала выполните кластеризацию")
            return

        try:
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            if self.current_features_type == "отобранные" and self.selected_features:
                data = self.encoded_df.values[:, self.selected_features]
            else:
                data = self.encoded_df.values

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            labels = self.labels
            unique_clusters = np.unique(labels)
            n_clusters = len(unique_clusters)

            if scaled_data.shape[1] > 2:
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(scaled_data)
                x_label = "PCA Component 1"
                y_label = "PCA Component 2"
            else:
                reduced_data = scaled_data
                x_label = "Feature 1"
                y_label = "Feature 2"

            self.figure = Figure(figsize=(8, 6), dpi=100)
            plot = self.figure.add_subplot(111)

            colors = plt.cm.get_cmap('viridis', n_clusters)
            for i, cluster in enumerate(unique_clusters):
                cluster_data = reduced_data[labels == cluster]
                plot.scatter(cluster_data[:, 0], cluster_data[:, 1],
                            color=colors(i),
                            label=f'Cluster {cluster}',
                            alpha=0.6)

            plot.set_title(f'Кластеры данных (всего {n_clusters} кластеров)')
            plot.set_xlabel(x_label)
            plot.set_ylabel(y_label)
            plot.legend()
            plot.grid(True)

            self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            self.show_error(f"Ошибка при построении графиков:\n{str(e)}")

    def update_data_labels(self):
        data_colors = {
            "исходные": "lightpink",
            "отобранные": "lightgreen",
            "обезличенные": "lightblue"
        }

        feature_colors = {
            "все": "lightyellow",
            "отобранные": "lightgreen"
        }

        self.data_label.config(
            text=f"Данные: {self.current_data_type}",
            bg=data_colors.get(self.current_data_type, "lightgray")
        )

        self.features_label.config(
            text=f"Признаки: {self.current_features_type}",
            bg=feature_colors.get(self.current_features_type, "lightgray")
        )

    def load_dataset(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV файлы", "*.csv"), ("Excel файлы", "*.xlsx;*.xls"), ("Все файлы", "*.*")]
        )
        if not file_path:
            return

        try:
            if file_path.endswith('.csv'):
                self.original_df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.original_df = pd.read_excel(file_path)
            else:
                raise ValueError("Неподдерживаемый формат файла")

            self.prepare_data()
            self.display_initial_data()
            self.current_data_type = "исходные"
            self.current_features_type = "все"
            self.update_data_labels()

        except Exception as e:
            self.show_error(f"Ошибка при загрузке файла:\n{str(e)}")

    def prepare_data(self):
        self.encoded_df = self.original_df.copy()

        if 'student_id' in self.encoded_df.columns:
            self.encoded_df = self.encoded_df.drop('student_id', axis=1)

        categorical_cols = self.encoded_df.select_dtypes(include=['object', 'category']).columns

        if not categorical_cols.empty:
            try:
                encoder = OneHotEncoder(drop='first', sparse_output=False)
            except TypeError:
                encoder = OneHotEncoder(drop='first', sparse=False)

            encoded_data = encoder.fit_transform(self.encoded_df[categorical_cols])
            encoded_cols = encoder.get_feature_names_out(categorical_cols)

            self.encoded_df = self.encoded_df.drop(categorical_cols, axis=1)
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=self.encoded_df.index)
            self.encoded_df = pd.concat([self.encoded_df, encoded_df], axis=1)

        self.reset_state()

    def anonymize_data(self):
        if self.original_df is None:
            messagebox.showwarning("Ошибка", "Сначала загрузите датасет")
            return

        try:
            self.anonymized_df = self.original_df.copy()

            if 'student_id' in self.anonymized_df.columns:
                self.anonymized_df['student_id'] = self.anonymized_df['student_id'].apply(
                    lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:8])

            categorical_cols = self.anonymized_df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if col != 'student_id':
                    self.anonymized_df[col] = self.anonymized_df[col].apply(
                        lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:8])

            self.prepare_anonymized_data()
            self.display_anonymized_data()
            self.current_data_type = "обезличенные"
            self.update_data_labels()

        except Exception as e:
            self.show_error(f"Ошибка при обезличивании данных:\n{str(e)}")

    def prepare_anonymized_data(self):
        if self.anonymized_df is None:
            return

        temp_df = self.anonymized_df.copy()

        if 'student_id' in temp_df.columns:
            temp_df = temp_df.drop('student_id', axis=1)

        categorical_cols = temp_df.select_dtypes(include=['object', 'category']).columns

        if not categorical_cols.empty:
            try:
                encoder = OneHotEncoder(drop='first', sparse_output=False)
            except TypeError:
                encoder = OneHotEncoder(drop='first', sparse=False)

            encoded_data = encoder.fit_transform(temp_df[categorical_cols])
            encoded_cols = encoder.get_feature_names_out(categorical_cols)

            temp_df = temp_df.drop(categorical_cols, axis=1)
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=temp_df.index)
            self.encoded_df = pd.concat([temp_df, encoded_df], axis=1)

        self.reset_state()

    def optimized_add_feature_selection(self):
        if self.encoded_df is None:
            messagebox.showwarning("Ошибка", "Сначала загрузите датасет")
            return

        num = simpledialog.askinteger(
            "Количество признаков",
            "Введите число признаков для отбора:",
            parent=self.root,
            minvalue=1,
            maxvalue=len(self.encoded_df.columns),
            initialvalue=self.num_features_to_select
        )

        if num is None:
            return

        self.num_features_to_select = num

        try:
            data = self.encoded_df.values
            feature_names = list(self.encoded_df.columns)
            n_features = data.shape[1]
            max_features_to_select = min(self.num_features_to_select, n_features)

            base_features_scores = {}
            selected_indices = []
            remaining_indices = list(range(n_features))

            for step in range(max_features_to_select):
                best_score = -np.inf
                best_feature_idx = None
                best_feature_name = None

                for feature_idx in remaining_indices:
                    feature_name = feature_names[feature_idx]
                    base_feature = self._get_base_feature(feature_name)

                    if base_feature in base_features_scores:
                        continue

                    current_features = selected_indices + [feature_idx]
                    X = data[:, current_features]

                    if not self._is_ohe_feature(feature_name):
                        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)

                    score = self._calculate_feature_score(X)

                    if score > best_score:
                        best_score = score
                        best_feature_idx = feature_idx
                        best_feature_name = feature_name

                if best_feature_idx is None:
                    break

                selected_indices.append(best_feature_idx)
                remaining_indices.remove(best_feature_idx)
                base_feature = self._get_base_feature(best_feature_name)
                base_features_scores[base_feature] = best_score

            self.selected_features = selected_indices
            self.feature_scores = base_features_scores
            self.current_features_type = "отобранные"
            self.update_data_labels()

            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, f"Отобрано {len(selected_indices)} наиболее значимых признаков:\n\n")

            selected_full_names = [self.encoded_df.columns[i] for i in selected_indices]
            scores = [base_features_scores[self._get_base_feature(f)] for f in selected_full_names]

            for i, (feature, score) in enumerate(zip(selected_full_names, scores), 1):
                self.text_area.insert(tk.END, f"{i}. {feature} (score={score:.3f})\n")

        except Exception as e:
            self.show_error(f"Ошибка при отборе признаков:\n{str(e)}")

    def _is_ohe_feature(self, feature_name):
        parts = feature_name.split('_')
        return len(parts) > 1 and not parts[-1].isdigit()

    def _get_base_feature(self, feature_name):
        return feature_name.split('_')[0] if self._is_ohe_feature(feature_name) else feature_name

    def _calculate_feature_score(self, X):
        if X.shape[0] < 5 or X.shape[1] == 0:
            return 0.0

        try:
            kmeans = KMeans(n_clusters=2, n_init=10)
            labels = kmeans.fit_predict(X)

            if len(np.unique(labels)) < 2:
                return 0.0

            return silhouette_score(X, labels, metric='chebyshev')
        except:
            return 0.0

    def run_isodata(self, use_selected_features=False, use_anonymized=False):
        if self.encoded_df is None:
            self.show_error("Сначала загрузите датасет")
            return

        try:
            params = self.get_isodata_parameters()

            if use_anonymized:
                if self.anonymized_df is None:
                    messagebox.showwarning("Ошибка", "Сначала обезличьте данные")
                    return
                self.prepare_anonymized_data()
                data = self.encoded_df.values
                self.current_data_type = "обезличенные"
                self.text_area.insert(tk.END, "\nИспользуются ОБЕЗЛИЧЕННЫЕ данные\n")
            else:
                data = self.encoded_df.values
                self.current_data_type = "исходные"
                self.text_area.insert(tk.END, "\nИспользуются ИСХОДНЫЕ данные\n")

            if use_selected_features:
                if not self.selected_features:
                    messagebox.showwarning("Ошибка", "Сначала выполните отбор признаков")
                    return
                data = data[:, self.selected_features]
                feature_names = [self._get_base_feature(self.encoded_df.columns[i])
                                 for i in self.selected_features]
                self.current_features_type = "отобранные"
                self.text_area.insert(tk.END, f"\nИспользуются ОТОБРАННЫЕ признаки:\n{', '.join(feature_names)}\n")
            else:
                self.current_features_type = "все"
                self.text_area.insert(tk.END, "\nИспользуются ВСЕ признаки\n")

            self.update_data_labels()

            scaler = StandardScaler()
            data = scaler.fit_transform(data)

            n_clusters, self.labels = self.isodata_algorithm(data, **params)
            self.display_clustering_results(n_clusters)

            self.show_cluster_plots()

        except Exception as e:
            self.show_error(f"Ошибка при кластеризации:\n{str(e)}")

    def get_isodata_parameters(self):
        return {
            'k': int(self.entries["Желаемое число кластеров:"].get()),
            'n_min': int(self.entries["Мин. точек в кластере:"].get()),
            'sigma_max': float(self.entries["Макс. дисперсия:"].get()),
            'l': float(self.entries["Мин. расстояние между кластерами:"].get()),
            'max_iter': int(self.entries["Макс. итераций:"].get())
        }

    def isodata_algorithm(self, data, k, n_min, sigma_max, l, max_iter):
        n_samples = data.shape[0]
        initial_k = k = max(2, min(k, n_samples))

        centroids = [data[np.random.randint(n_samples)]]
        for _ in range(1, k):
            distances = np.max(np.abs(data[:, np.newaxis] - np.array(centroids)), axis=2)
            farthest_idx = np.argmax(np.min(distances, axis=1))
            centroids.append(data[farthest_idx])
        centroids = np.array(centroids)

        labels = np.zeros(n_samples)

        for iteration in range(max_iter):
            distances = np.max(np.abs(data[:, np.newaxis] - centroids), axis=2)
            new_labels = np.argmin(distances, axis=1)

            if np.all(labels == new_labels) and iteration > 5:
                break

            labels = new_labels

            centroids = np.array([data[labels == i].mean(axis=0) for i in range(k) if (labels == i).sum() > 0])
            k = len(centroids)

            if k > 2:
                cluster_sizes = np.bincount(labels, minlength=k)
                to_keep = cluster_sizes >= n_min
                if sum(to_keep) >= 2:
                    centroids = centroids[to_keep]
                    k = len(centroids)
                    distances = np.max(np.abs(data[:, np.newaxis] - centroids), axis=2)
                    labels = np.argmin(distances, axis=1)

            if k < 2 * initial_k:
                new_centroids = []
                for i in range(k):
                    cluster_points = data[labels == i]
                    if len(cluster_points) > 2 * n_min:
                        variance = np.max(np.abs(cluster_points - centroids[i]), axis=0).mean()
                        if variance > sigma_max * 0.8:
                            direction = np.random.normal(0, 1, data.shape[1])
                            new_centroids.append(centroids[i] + 0.5 * sigma_max * direction)
                            new_centroids.append(centroids[i] - 0.5 * sigma_max * direction)

                if new_centroids:
                    centroids = np.vstack([centroids, new_centroids])
                    k = len(centroids)

            if k > 2:
                centroid_distances = np.max(np.abs(centroids[:, np.newaxis] - centroids), axis=2)
                np.fill_diagonal(centroid_distances, np.inf)

                min_dist = np.min(centroid_distances)
                while min_dist < l * 0.7 and k > 2:
                    i, j = np.unravel_index(np.argmin(centroid_distances), (k, k))
                    new_center = (centroids[i] + centroids[j]) / 2
                    centroids[i] = new_center
                    centroids = np.delete(centroids, j, axis=0)
                    k -= 1

                    if k <= 2:
                        break

                    centroid_distances = np.max(np.abs(centroids[:, np.newaxis] - centroids), axis=2)
                    np.fill_diagonal(centroid_distances, np.inf)
                    min_dist = np.min(centroid_distances)

        return k, labels

    def display_clustering_results(self, n_clusters):
        if n_clusters is None:
            self.show_error("Кластеризация не удалась")
            return

        if self.current_data_type == "обезличенные":
            result_df = self.anonymized_df.copy()
        else:
            result_df = self.original_df.copy()

        result_df['Cluster'] = self.labels

        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, result_df.to_string())
        self.cluster_label.config(text=f"Кластеры: {n_clusters}")

    def evaluate_clusters(self):
        if self.labels is None:
            messagebox.showwarning("Ошибка", "Сначала выполните кластеризацию")
            return

        try:
            if self.current_features_type == "отобранные" and self.selected_features:
                data = self.encoded_df.values[:, self.selected_features]
            else:
                data = self.encoded_df.values

            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            labels = self.labels

            if len(np.unique(labels)) < 2:
                self.quality_label.config(text="Качество: 0.0 (1 кластер)")
                return 0.0

            centroids = np.array([data[labels == i].mean(axis=0) for i in np.unique(labels)])

            intra_dists = []
            for i, center in zip(np.unique(labels), centroids):
                cluster_points = data[labels == i]
                dists = np.max(np.abs(cluster_points - center), axis=1)
                intra_dists.append(np.mean(dists))

            avg_intra_dist = np.mean(intra_dists) if intra_dists else 0

            inter_dists = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    dist = np.max(np.abs(centroids[i] - centroids[j]))
                    inter_dists.append(dist)

            min_inter_dist = np.min(inter_dists) if inter_dists else 0

            if avg_intra_dist > 0:
                quality_score = min_inter_dist / avg_intra_dist
            else:
                quality_score = float('inf')

            normalized_score = min(1.0, max(0.0, 1 - np.exp(-0.5 * quality_score)))

            quality_text = f"Качество ({self.current_data_type}, {self.current_features_type} признаки): {normalized_score:.3f}"
            self.quality_label.config(text=quality_text)
            self.text_area.insert(tk.END, f"\n{quality_text}\n")

            return normalized_score

        except Exception as e:
            self.show_error(f"Ошибка оценки качества: {str(e)}")
            return 0.0

    def display_initial_data(self):
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, self.original_df.to_string())
        self.cluster_label.config(text="Кластеры: 0")
        self.quality_label.config(text="Качество: -")

    def display_anonymized_data(self):
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, self.anonymized_df.to_string())
        self.cluster_label.config(text="Кластеры: 0")
        self.quality_label.config(text="Качество: -")

    def show_progress_dialog(self, title):
        progress = tk.Toplevel(self.root)
        progress.title(title)
        progress.geometry("400x100")

        tk.Label(progress, text=title).pack(pady=5)
        progress_bar = ttk.Progressbar(progress, orient=tk.HORIZONTAL, length=300, mode='determinate')
        progress_bar.pack(pady=5)

        self.progress_label = tk.Label(progress, text="")
        self.progress_label.pack()

        return progress

    def update_progress(self, dialog, current, total, message):
        dialog.children['!progressbar']['value'] = (current / total) * 100
        self.progress_label.config(text=message)
        dialog.update()

    def reset_state(self):
        self.selected_features = None
        self.labels = None
        self.feature_scores = {}

    def show_error(self, message):
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, message)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")
    app = DatasetViewer(root)
    root.mainloop()