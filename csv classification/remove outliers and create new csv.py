import pandas as pd
from pathlib import Path

def elimina_files_amb_outliers(df, factor_iqr=1.5, llindar=0.2):
    """
    Elimina files amb una proporció alta d'outliers en tots els atributs numèrics.
    
    :param df: DataFrame d'entrada.
    :param factor_iqr: Factor multiplicatiu per determinar els límits d'IQR (1.5 és el valor per defecte).
    :param llindar: Proporció màxima d'outliers permesa per fila (0.2 = 20%).
    :return: DataFrame filtrat.
    """
    # Còpia del DataFrame per no modificar l'original
    df_filtrat = df.copy()
    
    # Selecciona només columnes numèriques (exclou 'label')
    columnes_numeriques = df_filtrat.select_dtypes(include=['float64', 'int64']).columns
    
    # Inicialitza una columna per comptar outliers
    df_filtrat['num_outliers'] = 0
    
    # Calcula els límits d'IQR i compta outliers per fila
    for columna in columnes_numeriques:
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor_iqr * IQR
        upper_bound = Q3 + factor_iqr * IQR
        
        # Identifica outliers (valors fora dels límits) i suma'ls
        df_filtrat['num_outliers'] += ((df[columna] < lower_bound) | (df[columna] > upper_bound)).astype(int)
    
    # Calcula la proporció d'outliers per fila
    df_filtrat['proporcio_outliers'] = df_filtrat['num_outliers'] / len(columnes_numeriques)
    
    # Filtra les files amb una proporció d'outliers menor o igual al llindar
    df_filtrat = df_filtrat[df_filtrat['proporcio_outliers'] <= llindar]
    
    # Elimina les columnes temporals
    df_filtrat.drop(columns=['num_outliers', 'proporcio_outliers'], inplace=True)
    
    return df_filtrat

if __name__ == "__main__":

    # Directori actual del fitxer
    current_dir = Path(__file__).parent

    # Camins dels arxius CSV
    cami_csv_3s = current_dir.parent / "datasets" / "Data1" / "features_3_sec.csv"
    cami_csv_30s = current_dir.parent / "datasets" / "Data1" / "features_30_sec.csv"

    # Llegir dades
    data3s = pd.read_csv(cami_csv_3s)
    data30s = pd.read_csv(cami_csv_30s)

    # Filtrar files amb molts outliers
    llindar_outliers = 0.2  # Permetre un màxim del 20% d'outliers per fila
    factor_iqr = 1.0  # Incrementa la sensibilitat als outliers reduint el factor IQR

    print("Netejant dataset de 30s...")
    data30s_filtrat = elimina_files_amb_outliers(data30s, factor_iqr=factor_iqr, llindar=llindar_outliers)

    print("Netejant dataset de 3s...")
    data3s_filtrat = elimina_files_amb_outliers(data3s, factor_iqr=factor_iqr, llindar=llindar_outliers)

    # Guardar els fitxers filtrats
    output_3s = current_dir.parent / "datasets" / "dades_sense_outliers_3s.csv"
    output_30s = current_dir.parent / "datasets" / "dades_sense_outliers_30s.csv"

    data3s_filtrat.to_csv(output_3s, index=False)
    data30s_filtrat.to_csv(output_30s, index=False)

    # Mostra la distribució després de filtrar
    print("Distribució 3s després de filtrar:")
    print(data3s_filtrat['label'].value_counts())

    print("Distribució 30s després de filtrar:")
    print(data30s_filtrat['label'].value_counts())
