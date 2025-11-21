# 💰 Personal Finance Manager

A simple and interactive **Personal Finance Management System** built using **Python**, **SQLite**, and **Streamlit**.  
This application allows you to **add, view, update, delete, search, and visualize** your financial transactions easily — all through a clean web-based interface.

---

## 🚀 Features

✅ Add new transactions with details like ID, Type, Purpose, Amount, and Date.  
✅ View all transactions in a tabular format.  
✅ Update or delete existing transactions.  
✅ Search transactions by a specific date or date range.  
✅ View **Weekly and Monthly spending trends** using line charts.  
✅ Analyze **Purpose-wise spending** through bar charts.  
✅ Automatically stores all data in a local **SQLite database (finance.db)**.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** – for the interactive UI
- **SQLite3** – for local database storage
- **Pandas** – for data handling
- **Matplotlib** – for plotting trends

---

## 📂 Project Structure

📁 Financial Management System
│
├── main.py # Main Streamlit application
├── finance.db # SQLite database (auto-created)
├── README.md # Documentation (this file)
└── requirements.txt # Dependencies (optional)


---

## ⚙️ Installation & Setup

### 1️⃣ Clone or download this project
```bash
git clone https://github.com/yourusername/finance-manager.git
cd finance-manager

2️⃣ Install required dependencies

Make sure Python is installed. Then run:

pip install streamlit pandas matplotlib

▶️ How to Run

If your file is named main.py and is inside a folder like Main,
run the app using the following command:

streamlit run Main/main.py


💡 Tip: You must be inside your project’s root directory in the terminal before running the command.

Once you run it, Streamlit will open your Finance Manager web app in your browser at:

http://localhost:8501

📊 Features Overview
🔹 Add Transaction

Enter details such as Transaction Number, ID, Type (Credited/Debited), Purpose, Amount, and Date.

🔹 View Transactions

Displays all transactions in a table with total credited, total debited, and current balance.

🔹 Search by Date

Find transactions that occurred on a specific date or between two dates.

🔹 Update Transaction

Edit any field (Purpose, Amount, Date, etc.) for a specific transaction.

🔹 Delete Transaction

Remove an unwanted transaction safely.

🔹 View Trends

Visualizes your weekly and monthly spending patterns using line charts.

🔹 Purpose-wise Spending

Shows your expenses by category (Purpose) using a bar chart.

💡 Notes

The database file finance.db is automatically created in the same folder when you run the app for the first time.

You can rename or move it, but make sure to update its path in the code if you do.

To stop the app, press Ctrl + C in your terminal.

🧠 Future Enhancements

Add login authentication

Export data as CSV or Excel

Add savings goals and reminders

Dark mode and custom UI styling

👨‍💻 Author

Aarush Rahul Patel
📅 Built with passion for coding, finance, and data visualization
✨ “Code your money, manage your future.”

🪪 License

This project is open-source and free to use for learning and personal projects.
