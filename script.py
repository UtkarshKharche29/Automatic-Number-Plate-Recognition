import sqlite3

conn = sqlite3.connect('database.db')
print("Opened database successfully")

conn.execute(
    'CREATE TABLE plate_numbers (plate_no TEXT)')

conn.execute(
    'CREATE TABLE pass_application (first_name TEXT,last_name TEXT,email TEXT,plate_no TEXT PRIMARY KEY,address1 TEXT,address2 TEXT,city TEXT,state TEXT, zip INTEGER)')

conn.execute(
    'CREATE TABLE pass_allowed (first_name TEXT,last_name TEXT,email TEXT,plate_no TEXT PRIMARY KEY,address1 TEXT,address2 TEXT,city TEXT,state TEXT, zip INTEGER)')


conn.execute(
    'CREATE TABLE vehicle_problem (first_name TEXT,last_name TEXT,email TEXT,plate_no TEXT PRIMARY KEY,address1 TEXT,address2 TEXT,city TEXT,state TEXT, zip INTEGER)')

print("Table created successfully")

conn.close()
