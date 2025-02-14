import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import root_mean_squared_error, mean_squared_error

TARGET = "Price_in_euros"
FEATURES_NUM = ["Inches", "Ram_num", "Weight_num"]

# /----------------------------------------------------------------------------------/
# Función que lee el csv y lo carga en un dataframe
# /----------------------------------------------------------------------------------/
def read_csv(filename):
    result = pd.read_csv(filename, index_col=0)
    result.index.name = None
    return result

def split_categorica(dataframe, column, num = 2):
    result = column + "_split"
    dataframe[result] = ""
    for idx in range(0,num):
        dataframe[result] += dataframe[column].str.split(" ", expand=True)[idx]
    return result

def get_RSME(y, y_pred):
    RMSE = root_mean_squared_error(y, y_pred)
    return RMSE
    #return np.sqrt(mean_squared_error(y, y_pred))

# /----------------------------------------------------------------------------------/
# Función para realizar transformaciones sobre un dataframe, de tal manera que se puedan
# aplicar a cualquier set de datos (train, test...)
# /----------------------------------------------------------------------------------/
def generate_data(dataframe):

    # Transformamos object a variables numéricas
    #features_num = ["Inches", "Ram_num", "Weight_num"]

    # result["Inches_num"] = result["Inches"].astype(float)
    dataframe["Ram_num"] = dataframe['Ram'].str.replace('GB', '').astype(int)
    dataframe["Weight_num"] = dataframe['Weight'].str.replace('kg', '').astype(float)

    cat_columns = ["Company", "TypeName"]

    # Hacemos un split de los valores para obtener una nueva columna
    cat_columns.append(split_categorica(dataframe, "Cpu", 2))
    cat_columns.append(split_categorica(dataframe, "Gpu", 2))
    cat_columns.append(split_categorica(dataframe, "OpSys", 1))

    features_cat = [s + "_cat" for s in cat_columns]

    # Transformamos a ordinal las variables categoricas seleccionadas
    ord_encoder = OrdinalEncoder()
    dataframe[features_cat] = ord_encoder.fit_transform(dataframe[cat_columns]) + 1
    
    return FEATURES_NUM + features_cat

# /----------------------------------------------------------------------------------/
# Carga el dataframe y se crean las columnas transformadas
# /----------------------------------------------------------------------------------/
def get_dataframe(filename):
    result = read_csv(filename)
    features = generate_data(result)
    if TARGET in result.columns:
        features.append(TARGET)
    return result[features]


# /----------------------------------------------------------------------------------/
# Función que aplica a un dataframe todas las transformaciones identificadas
# /----------------------------------------------------------------------------------/
def transform(dataframe, logaritmo=True, std_scaler=True):
    
    result = dataframe.copy()

    # Aplico transformación logarítmica a estas variables
    if logaritmo:
        features_transform = ["Weight_num"] 
        for feature in features_transform:
            #print(f"Aplicando logaritmo a '{feature}'")
            result[feature] = result[feature].apply(np.log)

    if std_scaler:
        std_scaler = StandardScaler()
        #print(f"Escalando variables: {FEATURES_NUM}")
        result[FEATURES_NUM] = std_scaler.fit_transform(result[FEATURES_NUM])
    
    return result


# /----------------------------------------------------------------------------------/
# Función que devuelte los dataframes X e y 
# /----------------------------------------------------------------------------------/
def get_X_y(dataframe):
    features = dataframe.columns.to_list()

    if TARGET in features:
        features.remove(TARGET)

    X_set = dataframe[features]
    y_set = None

    if (TARGET in dataframe.columns):
        y_set = dataframe[TARGET]

    return X_set, y_set
