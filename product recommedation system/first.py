#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import mysql.connector
import hashlib
import base64
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


# 🔒 Initialize session state at the start
def init_session_state():
    defaults = {
        "logged_in": False,
        "username": "",
        "recent": [],
        "cart": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# In[3]:


def set_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        bg_image_style = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """
        st.markdown(bg_image_style, unsafe_allow_html=True)
    else:
        st.warning("Background image not found.")

set_background("C:\\Users\\priya\\Downloads\\basketimg.png")


# In[4]:


# ---------- MySQL Connection ----------
def create_connection():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="product_recommender"
        )
    except mysql.connector.Error as e:
        st.error(f"Database connection failed: {e}")
        return None


# In[ ]:


# ---------- Create Orders Table ----------
def create_orders_table():
    conn = create_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255),
            product_id VARCHAR(255),
            product_name TEXT,
            product_price FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        conn.close()

create_orders_table()


# In[ ]:


# Auth Helpers

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_user_exists(username):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def signup_user(username, password):
    if check_user_exists(username):
        return False
    conn = create_connection()
    if not conn:
        return False
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hash_password(password)))
    conn.commit()
    conn.close()
    return True


# In[ ]:


def login_user(username, password):
    user = check_user_exists(username)
    if user and user[2] == hash_password(password):
        st.session_state.logged_in = True
        st.session_state.username = username
        return True
    return False


# In[ ]:


def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.cart = []
    st.rerun()


# In[ ]:


# ---------- Load Product Data ----------
@st.cache_data
def load_data():
    data = pd.read_csv("marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv", sep='\t')
    data = data[['Product Id', 'Product Name', 'Product Rating', 'Product Image Url', 'Product Description', 'Product Tags']].fillna("")
    data['Price'] = (data.index % 10 + 1) * 10
    data["Product Rating"] = pd.to_numeric(data["Product Rating"], errors="coerce")
    return data

data = load_data()


# In[ ]:


# ---------- Recommendation Logic ----------
@st.cache_resource
def compute_similarity_matrix(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['Product Tags'] + " " + data['Product Description'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

similarity_matrix = compute_similarity_matrix(data)


# In[ ]:


def recommend_products(name, min_rating=3.0):
    idx = data[data['Product Name'].str.contains(name, case=False)].index
    if not len(idx): return []
    scores = sorted(list(enumerate(similarity_matrix[idx[0]])), key=lambda x: x[1], reverse=True)
    return [data.iloc[i[0]] for i in scores[1:] if data.iloc[i[0]]['Product Rating'] >= min_rating][:5]


# In[ ]:


# ---------- Save Order ----------
def save_to_db(product):
    conn = create_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO orders (username, product_id, product_name, product_price) VALUES (%s, %s, %s, %s)",
            (st.session_state.username, product['Product Id'], product['Product Name'], product['Price'])
        )
        conn.commit()
        conn.close()


# In[ ]:


# ---------- Pages ----------
def login_page():
    st.title("🔐 Login")
    choice = st.radio("", ["Login", "Sign Up"])
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")
    if st.button(choice):
        success = login_user(user, pw) if choice == "Login" else signup_user(user, pw)
        if success:
            st.rerun()
        else:
            st.error("Invalid login or signup failed.")


# In[ ]:


def home_page():
    st.title("🏠 Home")
    st.subheader("🔥 Trending Products")
    for product in data.head(5).to_dict('records'):
        st.image(product['Product Image Url'], width=150)
        st.markdown(f"""
        **{product['Product Name']}**
        ⭐ {product['Product Rating']} | 💰 ${product['Price']}
        📝 {product['Product Description']}
        """)
        if st.button("🛍 Buy Now", key=f"home_{product['Product Id']}"):
            st.session_state.cart.append(product)
            st.success("Added to cart!")


# In[ ]:


def top_rated_page():
    st.title("🌟 Top Rated Products")
    top_rated = data[data["Product Rating"] >= 4.5].sort_values(by="Product Rating", ascending=False).head(10)
    for product in top_rated.to_dict('records'):
        st.image(product['Product Image Url'], width=150)
        st.markdown(f"**{product['Product Name']}** - ⭐ {product['Product Rating']} - 💰 ${product['Price']}\n📝 {product['Product Description']}")
        if st.button("🛒 Add to Cart", key=f"toprated_{product['Product Id']}"):
            st.session_state.cart.append(product)
            st.success("Added to cart!")


# In[2]:


def recommend_page():
    st.title("🛍 Recommendations")
    name = st.text_input("Product Name")
    rating = st.slider("Min Rating", 1.0, 5.0, 3.0, 0.5)

    # Initialize cart and buy_now_product in session state if not already set
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'cart' not in st.session_state:
        st.session_state.cart = []
    if 'buy_now_product' not in st.session_state:
        st.session_state.buy_now_product = None

    # Get recommendations
    if st.button("Get Recommendations"):
        st.session_state.recommendations = recommend_products(name, rating)
        st.session_state.buy_now_product = None  # Clear previous buy_now_product

    # Display recommendations if available
    if 'recommendations' in st.session_state:
        for product in st.session_state.recommendations:
            st.image(product['Product Image Url'], width=150)
            st.markdown(
                f"**{product['Product Name']}** - ⭐ {product['Product Rating']} - 💰 ${product['Price']}\n\n"
                f"📝 {product['Product Description']}"
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(f"Add to Cart: {product['Product Name']}", key="cart_" + str(product['Product Id'])):
                    st.session_state.cart.append(product)
                    st.success("Added to cart")
            with col2:
                if st.button(f"Buy Now: {product['Product Name']}", key="buy_" + str(product['Product Id'])):
                    # Set the product to the buy_now_product session state when clicked
                    st.session_state.buy_now_product = product.to_dict() if hasattr(product, "to_dict") else product


    # Buy Now form (show only when a product is selected)
    if st.session_state.buy_now_product is not None:
        st.markdown("## 🧾 Purchase Details")
        st.image(st.session_state.buy_now_product['Product Image Url'], width=200)
        st.markdown(f"**{st.session_state.buy_now_product['Product Name']}** - 💰 ${st.session_state.buy_now_product['Price']}")
        st.write("DEBUG - Selected Product for Buy Now:", st.session_state.buy_now_product)


        # Name and address form fields
        buyer_name = st.text_input("Your Name")
        address = st.text_area("Delivery Address")

        # Handle the purchase confirmation
        if st.button("Confirm Purchase"):
            if buyer_name and address:
                # Simulate order processing
                st.success(f"Thank you, {buyer_name}! Your order for {st.session_state.buy_now_product['Product Name']} has been placed.")
                
                # Save the order to the database
                save_to_db(st.session_state.buy_now_product)

                # Reset the buy_now_product after order is placed
                st.session_state.buy_now_product = None  
            else:
                st.error("Please provide your name and address before confirming the purchase.")


# In[ ]:


def analytics_dashboard():
    st.title("📊 Analytics Dashboard")
    st.subheader("Average Rating by Product")
    avg_rating = data.groupby("Product Name")["Product Rating"].mean().sort_values(ascending=False).head(10)
    st.bar_chart(avg_rating)
    st.subheader("Rating Distribution")
    st.bar_chart(data["Product Rating"].value_counts().sort_index())


# In[ ]:


def cart_page():
    st.title("🛒 Cart")
    total = 0
    for item in st.session_state.cart:
        st.image(item['Product Image Url'], width=150)
        st.markdown(f"**{item['Product Name']}** - 💰 ${item['Price']}")
        total += item['Price']
    st.markdown(f"### Total: ${total}")
    if st.button("✅ Checkout"):
        for item in st.session_state.cart:
            save_to_db(item)
        st.session_state.cart.clear()
        st.success("Order placed")


# In[ ]:


def profile_page():
    st.title(f"👤 Profile - {st.session_state.username}")
    st.subheader("🧾 Order History")
    
    conn = create_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT product_name, product_price, created_at 
            FROM orders 
            WHERE username = %s 
            ORDER BY created_at DESC
        """, (st.session_state.username,))
        orders = cursor.fetchall()
        conn.close()

        if orders:
            for name, price, date in orders:
                st.markdown(f"**{name}** - 💰 ${price} on 📅 {date.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("No orders found.")
    else:
        st.error("Could not connect to the database.")


# In[ ]:


def search_page():
    st.title("🔎 Search Products")
    query = st.text_input("Search by name or tag")
    rating = st.slider("Minimum Rating", 1.0, 5.0, 3.0, 0.5)
    filtered = data[
        data["Product Name"].str.contains(query, case=False) |
        data["Product Tags"].str.contains(query, case=False)
    ]
    filtered = filtered[filtered["Product Rating"] >= rating]
    for product in filtered.to_dict('records'):
        st.image(product['Product Image Url'], width=150)
        st.markdown(f"**{product['Product Name']}** - ⭐ {product['Product Rating']} - 💰 ${product['Price']}\n📝 {product['Product Description']}")
        if st.button(f"Add to Cart", key=f"search_{product['Product Id']}"):
            st.session_state.cart.append(product)
            st.success("Added to cart!")


# In[ ]:


# ---------- Add Product Page ----------
def add_product_page():
    st.title("📦 Add New Product")
    with st.form("add_product_form"):
        product_id = st.text_input("Product ID")
        name = st.text_input("Product Name")
        rating = st.slider("Product Rating", 1.0, 5.0, 3.0, 0.5)
        description = st.text_area("Product Description")
        tags = st.text_input("Product Tags")
        price = st.number_input("Price", min_value=0.0, step=1.0)
        image_file = st.file_uploader("Upload Product Image", type=["jpg", "jpeg", "png"])

        submitted = st.form_submit_button("Add Product")
        if submitted:
            if not all([product_id, name, description, tags, price, image_file]):
                st.warning("Please fill all fields and upload an image.")
            else:
                # Save image to static/images
                os.makedirs("static/images", exist_ok=True)
                image_path = os.path.join("static/images", image_file.name)
                with open(image_path, "wb") as f:
                    f.write(image_file.getbuffer())
                image_url = f"static/images/{image_file.name}"

                # Append to TSV file
                new_row = pd.DataFrame([{
                    'Product Id': product_id,
                    'Product Name': name,
                    'Product Rating': rating,
                    'Product Image Url': image_url,
                    'Product Description': description,
                    'Product Tags': tags,
                    'Price': price
                }])

                new_row.to_csv("marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv", mode='a', sep='\t', header=False, index=False)
                st.success("✅ Product added successfully!")


# In[ ]:


# ---------- Main App ----------
if not st.session_state.logged_in:
    login_page()
else:
    page = st.sidebar.radio("Go to:", [
        "🏠 Home",
        "🌟 Top Rated",
        "🛍 Recommendations",
        "🔎 Search",
        "📊 Analytics",
        "🛒 Cart",
        "👤 Profile",
        "📦 Add Product",     
        "🔓 Logout"
    ])

    if page == "🏠 Home":
        home_page()
    elif page == "🌟 Top Rated":
        top_rated_page()
    elif page == "🛍 Recommendations":
        recommend_page()
    elif page == "🔎 Search":
        search_page()
    elif page == "📊 Analytics":
        analytics_dashboard()
    elif page == "🛒 Cart":
        cart_page()
    elif page == "👤 Profile":
        profile_page()
    elif page == "📦 Add Product":
        add_product_page()
    elif page == "🔓 Logout":
        logout()


