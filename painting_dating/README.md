FROM https://github.com/valer1435/painting_dating with exception of Explanation.ipynb, which is my attempt to reproduce the results.


### Введение
Данный проект был написан в рамках хакатона, проводимого летней школой СЛОН http://school-slon.ru/. В рамках выполнения задания была построена модель для определения временной принадлежности картины/фрески к тому или иному историческому периоду.


### Описание алгоритма
Данные брались с сайта
https://www.wga.hu
Для большего баланса классов было решено объединить период 201-1300 гг  
![](https://github.com/valer1435/painting_dating/blob/master/README/data.png)  
Для решения задачи была использована следующая модель:
![](https://github.com/valer1435/painting_dating/blob/master/README/model_architecture.png)  
- Сеть vgg19 до слоя conv_5_1 (1)
- Преобразование выходного слоя (1) в Грамоподобную матрицу (2)
- Отбор 8000 фич из (2)
- SVC c 13 классами


# Результаты

На тестовой выборке модель показала следующие результаты:
- MSE: 4  
- F1-score: 0.5  
Confusion matrix:
 ![](https://github.com/valer1435/painting_dating/blob/master/README/results.png)  
