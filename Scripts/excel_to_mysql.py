import os
import pandas as pd
import mysql.connector
from pandas import ExcelFile

# MySQL database connection settings
db_config = {
    'host': 'localhost',
    'user': 'cognitus',
    'password': 'student',
    'database': 'generative_mapping'
}

# Path to the folder containing Excel files
excel_folder = '/home/ubuntu/generative_mapping/generative_mapping/Data'

# List of all Excel files in the folder
excel_files = [file for file in os.listdir(excel_folder) if file.endswith('.xlsx')]

# Connect to the MySQL database
db_connection = mysql.connector.connect(**db_config)
cursor = db_connection.cursor()

# Looping through each Excel file and insert data into the database
for excel_file in excel_files:
    file_path = os.path.join(excel_folder, excel_file)
    
    try:
        xls = ExcelFile(file_path)
        # Assuming only one sheet in the Excel file; you can modify this if needed
        df = xls.parse(xls.sheet_names[0])
        # Performing operations on the DataFrame 'df' here
    except Exception as e:
        print(f"Error reading {excel_file}: {str(e)}")
    
    table_name = excel_file[:-5]  # Removing ".xlsx" extension for table name

    create_table_query = f"CREATE TABLE IF NOT EXISTS `{table_name}` ("
    create_table_query += ', '.join([f'`{col}` LONGTEXT' for col in df.columns])
    create_table_query += ")"
    cursor.execute(create_table_query)
    db_connection.commit()

    print(f"Table '{table_name}' creating and data inserting.")

    # Inserting data row by row
    for index, row in df.iterrows():
        # Replacing 'nan' values with None (NULL in MySQL)
        row = [None if pd.isna(value) else value for value in row]

        insert_query = f"INSERT INTO `{table_name}` ({', '.join([f'`{col}`' for col in df.columns])}) VALUES ({', '.join(['%s' for _ in df.columns])})"
        cursor.execute(insert_query, tuple(row))
        db_connection.commit()

    print(f"Table '{table_name}' created and data inserted.")

cursor.close()
db_connection.close()
