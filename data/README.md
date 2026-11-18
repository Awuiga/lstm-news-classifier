# AG News

Проект использует AG's News Topic Classification Dataset: 120 000 записей в исходном train и 7600 в test. Версия классификационного набора подготовлена Xiang Zhang; описание связано с работой Xiang Zhang, Junbo Zhao и Yann LeCun, *Character-level Convolutional Networks for Text Classification*, 2015.

Источник файлов для воспроизведения — [AG News CSV в CharCnn_Keras](https://github.com/mhjabreel/CharCnn_Keras/tree/555590db4219b1243abb1918effd6a7425a2d75f/data/ag_news_csv). После нормализации CRLF → LF локальные файлы совпали с зеркалом:

| Файл | Git blob SHA-1 |
| --- | --- |
| `train.csv` | `694aa19971315410e000404523796b76349d3c1d` |
| `test.csv` | `130b40c3329c993468b8ed8a22c331315e478e0a` |

Сохраните оба CSV рядом с этим README. В них нет строки заголовка; столбцы — `label,title,description`, метки 1–4. Команды скачивания приведены в основном README.

[Описание происхождения и условий использования](https://github.com/mhjabreel/CharCnn_Keras/blob/555590db4219b1243abb1918effd6a7425a2d75f/data/ag_news_csv/readme.txt) указывает исследовательские и другие некоммерческие цели. Это не утверждение о лицензии MIT или Apache на тексты новостей. Сам корпус и веса модели в данном репозитории не распространяются. Условия исходных данных не подменяют лицензию пользовательского кода.
