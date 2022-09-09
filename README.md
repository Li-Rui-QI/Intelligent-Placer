# Intelligent-Placer
## Постановка задачи
Требуется создать “Intelligent Placer”:по поданной на вход фотографии нескольких предметов（3-5 предметов）на светлой горизонтальной поверхности и многоугольнику определяется, можно ли расположить одновременно все эти предметы на плоскости , чтобы они поместились в заданный многоугольник.

## Вход и выход
Вход: Фотография в формате jpg
Выход: Ответ в текстовом формате, записанный в файл result_<имя входной фотографии>.txt:
"True" - предметы можно поместить в многоугольник;
"Fasle" - предметы нельзя поместить в многоугольник.

## Требования к входным данным
Фотография：
1. Освещение равномерное.
2. фото должно иметь расширение .png или .jpg.
3. Разрешение от 5 Мп.
4 Фотография делается вертикально.
5 фон фото является белым листом.

Предметы:
1. Предметы не должны выходить за границы кадра.
2. предметы не должны перекрывать друг друга.
3. один предмет может присутствовать на фото лишь 1 раз.
4. предметы могут иметь разные ориентации/направление.

Многоугольнику:
1. количество ребер многоугольника должно быть явно различимо.
2. Число вершин многоугольника не более 10.
