import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def report_missing_values(df, name):
    print(f"\n--- Valores Faltantes en: {name} ---")
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    summary = pd.DataFrame({
        'Columnas': df.columns,
        'Valores Faltantes': missing_count,
        'Porcentaje (%)': missing_percent.round(2)
    }).sort_values(by='Valores Faltantes', ascending=False)
    print(summary[summary['Valores Faltantes'] > 0])
    return summary

# TU FUNCIÓN ORIGINAL EXACTA
def escaneo_profundo_y_union(ruta_raiz):
    lista_final = []
    conteo_drone = 0
    conteo_imu = 0
    conteo_tamper = 0
    ignorados = []
    
    cols_drone = {'date', 'time', 'lat', 'lon', 'alt', 'x_gyro', 'y_gyro', 'z_gyro', 'x_acc', 'y_acc', 'z_acc'}
    cols_imu = {'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label', 'flight_mode', 'maneuver', 'action', 'mode'}
    cols_tamper = {'latitude', 'longitude', 'altitude', 'pitch', 'roll', 'yaw', 'is_tampered', 'tampered', 'label', 'attack', 'status'}

    print(f"Iniciando escaneo de carpetas en: {ruta_raiz}...")

    for root, dirs, files in os.walk(ruta_raiz):
        for archivo in files:
            ext = os.path.splitext(archivo)[1].lower()
            
            # IGNORAR ARCHIVOS DE METADATOS Y ARCHIVOS OCULTOS
            if "meta" in archivo.lower() or archivo.startswith("."):
                continue
                
            if ext in [".csv", ".json"]:
                ruta_completa = os.path.join(root, archivo)
                subcarpeta_padre = os.path.basename(root)
                
                try:
                    if ext == ".csv":
                        df_temp = pd.read_csv(ruta_completa)
                    else:
                        df_temp = pd.read_json(ruta_completa)

                    if df_temp.empty or not isinstance(df_temp, pd.DataFrame):
                        continue

                    columnas_actuales = set(map(str.lower, df_temp.columns))
                    
                    categoria = "Sin Clasificar"
                    if "IncludeTakeoff" in ruta_completa:
                        categoria = "IncludeTakeoff"
                    elif "WithoutTakeoff" in ruta_completa:
                        categoria = "WithoutTakeoff"

                    df_temp['Sensor Type'] = os.path.splitext(archivo)[0]
                    df_temp['Archivo_Origen'] = archivo
                    df_temp['Categoria_Vuelo'] = categoria
                    df_temp['Subcarpeta_Origen'] = subcarpeta_padre
                    
                    interseccion_drone = cols_drone.intersection(columnas_actuales)
                    interseccion_imu = cols_imu.intersection(columnas_actuales)
                    interseccion_tamper = cols_tamper.intersection(columnas_actuales)

                    if len(interseccion_drone) >= 4:
                        df_temp['Tipo_Dataset'] = 'Regresion_Trayectoria'
                        lista_final.append(df_temp)
                        conteo_drone += 1
                    elif len(interseccion_imu) >= 2 or subcarpeta_padre.lower() in ['hover', 'ascent', 'descent', 'turn_left', 'turn_right', 'turn']:
                        df_temp['Tipo_Dataset'] = 'Clasificacion_Multiclase_IMU'
                        if not any(c in df_temp.columns for c in ['maneuver', 'label', 'action', 'mode']):
                            df_temp['maneuver'] = subcarpeta_padre
                        lista_final.append(df_temp)
                        conteo_imu += 1
                    elif len(interseccion_tamper) >= 2 or any(kw in subcarpeta_padre.lower() for kw in ['tamper', 'jamming', 'balanced', 'strong', 'subtle']):
                        df_temp['Tipo_Dataset'] = 'Clasificacion_Binaria_Tampering'
                        if not any(c in df_temp.columns for c in ['is_tampered', 'tampered', 'label', 'attack']):
                            df_temp['is_tampered'] = 1 if any(kw in subcarpeta_padre.lower() for kw in ['tamper', 'jamming', 'strong', 'subtle', 'balanced']) else 0
                        lista_final.append(df_temp)
                        conteo_tamper += 1
                    else:
                        df_temp['Tipo_Dataset'] = 'General'
                        lista_final.append(df_temp)
                        conteo_drone += 1
                
                except Exception as e:
                    ignorados.append(f"{ruta_completa} (Error: {e})")

    print(f"\n--- RESUMEN DE LECTURA DE DATASETS ---")
    print(f"- Archivos Dataset 1 (Regresión Trayectoria) : {conteo_drone}")
    print(f"- Archivos Dataset 2 (Multiclase IMU Log)    : {conteo_imu}")
    print(f"- Archivos Dataset 3 (Binario Tampering)      : {conteo_tamper}")
    
    if ignorados:
        print(f"- Archivos ignorados o no válidos: {len(ignorados)}")

    if not lista_final:
        print(f"No se encontraron archivos procesables en {ruta_raiz}.")
        return None
    
    print("Uniendo datos en RAM...")
    df_global = pd.concat(lista_final, ignore_index=True, sort=False)
    del lista_final
    return df_global

def cargar_dataset(opcion, ruta_base):
    # Lógica condicional: Si no existe la subcarpeta específica, escanea la raíz general
    if opcion == 1:
        ruta = os.path.join(ruta_base, 'DroneTrajectoryData')
        if not os.path.exists(ruta): ruta = ruta_base # Fallback a ruta general
            
        df = pd.read_csv("datos_consolidados_drones.csv") if os.path.exists("datos_consolidados_drones.csv") else escaneo_profundo_y_union(ruta)
        
        if df is None: raise FileNotFoundError(f"[ERROR] No se encontraron archivos en: {ruta}")
        if not os.path.exists("datos_consolidados_drones.csv"): 
            df[df['Tipo_Dataset'] == 'Regresion_Trayectoria'].to_csv("datos_consolidados_drones.csv", index=False)
        
        cols_X = ['x_gyro', 'y_gyro', 'z_gyro', 'x_acc', 'y_acc', 'z_acc', 'pitch', 'yaw', 'roll', 'wind_speed', 'wind_direction']
        cols_Y = ['north', 'east', 'down']
        df_clean = df.dropna(subset=cols_X + cols_Y).copy()
        
        X_scaled = StandardScaler().fit_transform(df_clean[cols_X].values)
        Y_scaled = StandardScaler().fit_transform(df_clean[cols_Y].values)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.4, random_state=42)
        return X_train, X_test, y_train, y_test, 'regresion', 3 

    elif opcion == 2:
        ruta = os.path.join(ruta_base, 'DroneFlightIMUSensorLog')
        if not os.path.exists(ruta): ruta = ruta_base 
            
        df = pd.read_csv("datos_consolidados_imu.csv") if os.path.exists("datos_consolidados_imu.csv") else escaneo_profundo_y_union(ruta)
        
        if df is None: raise FileNotFoundError(f"[ERROR] No se encontraron archivos en: {ruta}")
        if not os.path.exists("datos_consolidados_imu.csv"): 
            df[df['Tipo_Dataset'] == 'Clasificacion_Multiclase_IMU'].to_csv("datos_consolidados_imu.csv", index=False)
        
        target_col = 'maneuver' if 'maneuver' in df.columns else 'Subcarpeta_Origen'
        #ignore_cols = [target_col, 'Tipo_Dataset', 'Sensor Type', 'Sensor_Type', 'Archivo_Origen', 'Categoria_Vuelo', 'Subcarpeta_Origen']
        ignore_cols = [target_col, 'Tipo_Dataset', 'Sensor Type', 'Sensor_Type', 'Archivo_Origen', 'Categoria_Vuelo', 'Subcarpeta_Origen', 'label', 'action', 'mode', 'flight_mode']
        df_num = df.drop(columns=[c for c in ignore_cols if c in df.columns]).select_dtypes(include=[np.number])
        
        df_clean = df.dropna(subset=list(df_num.columns) + [target_col]).copy()
        X_scaled = StandardScaler().fit_transform(df_clean[list(df_num.columns)].values)
        
        Y_onehot = OneHotEncoder(sparse_output=False).fit_transform(df_clean[target_col].astype(str).values.reshape(-1, 1))
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_onehot, test_size=0.4, random_state=42)
        return X_train, X_test, y_train, y_test, 'multiclase', Y_onehot.shape[1] 

    elif opcion == 3:
        ruta = os.path.join(ruta_base, 'DroneTelemetryTampering')
        if not os.path.exists(ruta): ruta = ruta_base 
            
        df = pd.read_csv("datos_consolidados_tampering.csv") if os.path.exists("datos_consolidados_tampering.csv") else escaneo_profundo_y_union(ruta)
        
        if df is None: raise FileNotFoundError(f"[ERROR] No se encontraron archivos en: {ruta}")
        if not os.path.exists("datos_consolidados_tampering.csv"): 
            df[df['Tipo_Dataset'] == 'Clasificacion_Binaria_Tampering'].to_csv("datos_consolidados_tampering.csv", index=False)
        
        target_col = 'is_tampered'
        if target_col not in df.columns and 'Subcarpeta_Origen' in df.columns:
            df[target_col] = df['Subcarpeta_Origen'].astype(str).str.lower().apply(lambda v: 1 if any(kw in v for kw in ['tamper', 'attack', 'jam']) else 0)
            
        ignore_cols = [target_col, 'Tipo_Dataset', 'Sensor Type', 'Archivo_Origen', 'Categoria_Vuelo', 'Subcarpeta_Origen']
        df_num = df.drop(columns=[c for c in ignore_cols if c in df.columns]).select_dtypes(include=[np.number])
        
        cols_X = [c for c in df_num.columns if df_num[c].isnull().sum() < (0.7 * len(df))]
        X_df = df[cols_X].fillna(df[cols_X].mean()).fillna(0)
        X_scaled = StandardScaler().fit_transform(X_df.values)
        
        Y_raw = df[target_col].fillna(0).values.astype(int).reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_raw, test_size=0.4, random_state=42)
        return X_train, X_test, y_train, y_test, 'binaria', 1 

    else:
        raise ValueError("Opción no válida")