import sqlite3

conn = sqlite3.connect('database.db')
print("Opened database successfully")

conn.execute(
    'CREATE TABLE plate_numbers (plate_no TEXT)')
print("Table created successfully")
conn.close()
