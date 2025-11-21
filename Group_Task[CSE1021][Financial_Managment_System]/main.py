import sqlite3
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ DATABASE SETUP ------------------------
con = sqlite3.connect("finance.db")
cur = con.cursor()

cur.execute('''CREATE TABLE IF NOT EXISTS expenses (
    TransactionNumber INTEGER PRIMARY KEY,
    Transaction_ID TEXT,
    Transaction_Type TEXT,
    Purpose TEXT,
    Amount REAL,
    Date TEXT
)''')
con.commit()

# ------------------------ APP TITLE ------------------------
st.set_page_config(page_title="💰 Finance Manager", layout="centered")
st.title("💰 Personal Finance Manager")
st.markdown("Manage your transactions easily — add, view, update, or delete.")

# ------------------------ HELPER FUNCTIONS ------------------------
def get_data():
    df = pd.read_sql_query("SELECT * FROM expenses", con)
    return df

def add_transaction(tno,tid,ttype, tpurpose, tamount, tdate):
    cur.execute(
        "INSERT INTO expenses VALUES (?, ?, ?, ?, ?, ?)",
        (tno,tid, ttype, tpurpose, tamount, tdate)
    )
    con.commit()

def delete_transaction(tno):
    cur.execute("DELETE FROM expenses WHERE TransactionNumber = ?", (tno,))
    con.commit()
    return cur.rowcount

def update_transaction(tno, field, new_value):
    cur.execute(f"UPDATE expenses SET {field} = ? WHERE TransactionNumber = ?", (new_value, tno))
    con.commit()
    return cur.rowcount

def search_by_date(start_date, end_date=None):
    if end_date:
        query = "SELECT * FROM expenses WHERE Date BETWEEN ? AND ?"
        df = pd.read_sql_query(query, con, params=(start_date, end_date))
    else:
        query = "SELECT * FROM expenses WHERE Date = ?"
        df = pd.read_sql_query(query, con, params=(start_date,))
    return df
# ------------------------ SIDEBAR MENU ------------------------
menu = st.sidebar.radio(
    "Menu",
    ["Add Transaction", "View Transactions", "Search by Date", "Update Transaction", "Delete Transaction","View Trends","Purpose-wise Spending"]
)

# ------------------------ ADD TRANSACTION ------------------------
if menu == "Add Transaction":
    st.subheader("Add New Transaction")

    with st.form("add_form"):
        tno = st.number_input("Transaction Number", min_value=1, step=1)
        tid = st.number_input("Transaction ID", min_value=100000000000, max_value=999999999999)
        ttype = st.selectbox("Transaction Type", ["Credited", "Debited"])
        tpurpose = st.text_input("Purpose")
        tamount = st.number_input("Amount", min_value=0.0, step=0.01)
        tdate = st.date_input("Date")

        submitted = st.form_submit_button("Add Transaction")

        if submitted:
            try:
                add_transaction(tno,tid, ttype, tpurpose, tamount, tdate)
                st.success(f"✅ Transaction {tno} added successfully!")
            except sqlite3.IntegrityError:
                st.error("⚠️ Transaction Number already exists. Please use a unique number.")

# ------------------------ VIEW TRANSACTIONS ------------------------
elif menu == "View Transactions":
    st.subheader("All Transactions")

    df = get_data()
    if df.empty:
        st.info("No transactions found.")
    else:
        st.dataframe(df, use_container_width=True)
        total_credit = df[df["Transaction_Type"] == "Credited"]["Amount"].sum()
        total_debit = df[df["Transaction_Type"] == "Debited"]["Amount"].sum()
        balance = total_credit - total_debit

        st.markdown("---")
        st.metric("💵 Total Credited", f"₹{total_credit:.2f}")
        st.metric("💸 Total Debited", f"₹{total_debit:.2f}")
        st.metric("🧾 Current Balance", f"₹{balance:.2f}")

elif menu == "Search by Date":
    st.header("🔍 Search Transactions by Date")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date (optional)", value=None)

    if st.button("Search"):
        df = search_by_date(str(start_date), str(end_date) if end_date else None)
        if not df.empty:
            st.dataframe(df)
        else:
            st.warning("⚠️ No transactions found for the selected date(s).")
# ------------------------ UPDATE TRANSACTION ------------------------
elif menu == "Update Transaction":
    st.subheader("Update Transaction")

    df = get_data()
    if df.empty:
        st.info("No transactions available to update.")
    else:
        tno = st.selectbox("Select Transaction Number", df["TransactionNumber"].tolist())
        field = st.selectbox("Field to Update", ["Transaction_ID","Transaction_Type", "Purpose", "Amount", "Date"])

        if field == "Transaction_ID":
            new_value = st.number_input("New Transaction_ID", min_value=100000000000, max_value=999999999999)
        elif field == "Transaction_Type":
            new_value = st.selectbox("New Type", ["Credited", "Debited"])
        elif field == "Amount":
            new_value = st.number_input("New Amount", min_value=0.0, step=0.01)
        elif field == "Date":
            new_value = st.date_input("New Date")
        else:
            new_value = st.text_input("New Value")

        if st.button("Update"):
            rows = update_transaction(tno, field, new_value)
            if rows:
                st.success(f"✅ Transaction {tno} updated successfully!")
            else:
                st.warning("⚠️ Transaction not found.")

# ------------------------ DELETE TRANSACTION ------------------------
elif menu == "Delete Transaction":
    st.subheader("Delete Transaction")

    df = get_data()
    if df.empty:
        st.info("No transactions available to delete.")
    else:
        tno = st.selectbox("Select Transaction to Delete", df["TransactionNumber"].tolist())

        if st.button("Delete"):
            rows = delete_transaction(tno)
            if rows:
                st.success(f"✅ Transaction {tno} deleted successfully!")
            else:
                st.warning("⚠️ No record found.")

elif menu == "View Trends":
    st.header("📈 Weekly & Monthly Spending Trends")
    df = get_data()

    if df.empty:
        st.info("No data available.")
    else:
        # Ensure Date column is datetime
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])  # remove rows with invalid dates
        debited = df[df["Transaction_Type"] == "Debited"]

        if debited.empty:
            st.warning("⚠️ No 'Debited' transactions found to show trends.")
        else:
            # Weekly Spending Trend
            debited["Week"] = debited["Date"].dt.isocalendar().week
            weekly = debited.groupby("Week")["Amount"].sum()

            if weekly.empty:
                st.warning("⚠️ No weekly data available to plot.")
            else:
                fig, ax = plt.subplots()
                weekly.plot(ax=ax, marker="o", color="red")
                ax.set_title("Weekly Spending Trend")
                ax.set_xlabel("Week Number")
                ax.set_ylabel("Total Spent")
                st.pyplot(fig)

            # Monthly Spending Trend
            debited["Month"] = debited["Date"].dt.to_period("M")
            monthly = debited.groupby("Month")["Amount"].sum()

            if monthly.empty:
                st.warning("⚠️ No monthly data available to plot.")
            else:
                fig2, ax2 = plt.subplots()
                monthly.plot(ax=ax2, marker="o", color="blue")
                ax2.set_title("Monthly Spending Trend")
                ax2.set_xlabel("Month")
                ax2.set_ylabel("Total Spent")
                st.pyplot(fig2)


elif menu == "Purpose-wise Spending":
    st.header("🎯 Category/Purpose-wise Spending")
    df = get_data()

    if df.empty:
        st.info("No data available.")
    else:
        debited = df[df["Transaction_Type"] == "Debited"]
        purpose_sum = debited.groupby("Purpose")["Amount"].sum()

        if purpose_sum.empty:
            st.warning("⚠️ No 'Debited' transactions found to plot.")
        else:
            fig, ax = plt.subplots()
            purpose_sum.plot(kind="bar", ax=ax, color="orange")
            ax.set_title("Purpose-wise Spending")
            ax.set_xlabel("Purpose")
            ax.set_ylabel("Total Spent")
            plt.xticks(rotation=45)
            st.pyplot(fig)

