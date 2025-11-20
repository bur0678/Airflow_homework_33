from datetime import datetime
import dill
import pandas as pd
import os


def predict():
    os.chdir('..')
    path = os.getcwd()
    path = os.environ.get('PROJECT_PATH', '\\opt\\airflow\\plugins')
    model_filename = os.listdir(path=(f'{path}\\data\\models'))

    with open(f'{path}\\data\\models\\{model_filename[-1]}', 'rb') as input_file:
        model_to_load = dill.load(input_file)


    path_2 = f'{path}\\data\\test'
    test_file_list = os.listdir(path=(path_2))

    preds = pd.DataFrame(columns=['car_id', 'pred'])

    for i in test_file_list:
        df = pd.read_json(f'{path_2}\\{i}', typ='series')
        X_test = df.to_frame().T
        y_pred = model_to_load.predict(X_test)
        df = pd.DataFrame({'car_id':df.id, 'pred':y_pred})
        preds = pd.concat([preds, df], axis=0)

    preds.to_csv(f'data\\predictions\\{datetime.now().strftime('%d%m%Y_%H_%M')}.csv', index=False)

if __name__ == '__main__':
    predict()
