#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import mysql.connector
import hashlib
import base64


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[5]:


def set_background(image_path):
    """Set background image in Streamlit."""
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

# ✅ Call the function with your image path
set_background("C:\\Users\\priya\\Downloads\\basketimg.png")
# --- Initialize session state ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "recent" not in st.session_state:
    st.session_state["recent"] = []
if "cart" not in st.session_state:
    st.session_state["cart"] = []


# In[6]:


# --- MySQL Database Connection ---
def create_connection():
    return mysql.connector.connect(
        host="localhost",   # e.g., "localhost"
        user="root",   # e.g., "root"
        password="",
        database="product_recommender"
    )


# In[7]:


# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# In[9]:


# Function to check if user exists
def check_user_exists(username):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    conn.close()
    return user


# In[10]:


def signup_user(username, password):
    if check_user_exists(username):
        return False
    conn = create_connection()
    cursor = conn.cursor()
    hashed_pw = hash_password(password)
    cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hashed_pw))
    conn.commit()
    conn.close()
    return True


# In[11]:


# Function to login user
def login_user(username, password):
    user = check_user_exists(username)
    if user and user[2] == hash_password(password):  # user[2] is password_hash
        st.session_state["logged_in"] = True
        st.session_state["username"] = username
        return True
    return False


# In[12]:


# Function to logout
def logout():
    st.session_state["logged_in"] = False
    st.session_state.pop("username", None)


# In[13]:


@st.cache_data
def load_data():
    data = pd.read_csv("marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv", sep='\t')
    data = data[['Product Id', 'Product Name', 'Product Rating', 'Product Image Url', 'Product Description', 'Product Tags']]
    data.fillna("", inplace=True)
    data["Product Rating"] = pd.to_numeric(data["Product Rating"], errors="coerce")  # Ensure numeric
     # Add dummy price column
    data["Price"] = (data.index % 10 + 1) * 10  # Example prices: 10, 20, ..., 100
    return data

# ✅ Load it into a variable
data = load_data()


# In[14]:


# Compute TF-IDF similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['Product Tags'] + " " + data['Product Description'])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[15]:


def recommend_products(product_name, num_recommendations=5, min_rating=0):
    idx = data[data['Product Name'].str.contains(product_name, case=False, na=False)].index
    if len(idx) == 0:
        return []
    idx = idx[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in scores[1:]:
        product = data.iloc[i[0]]
        if product['Product Rating'] >= min_rating:
            recommendations.append(product)
        if len(recommendations) == num_recommendations:
            break
    return recommendations


# In[16]:


# --- Cart Functions ---
def add_to_cart(product):
    st.session_state["cart"].append(product)
    st.success(f"🛒 '{product['Product Name']}' added to cart!")


# In[17]:


def remove_from_cart(product_id):
    st.session_state["cart"] = [p for p in st.session_state["cart"] if p["Product Id"] != product_id]
    st.success("✅ Removed from cart")


# In[18]:


def save_cart_to_db(username, product):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO orders (username, product_id, product_name, product_price)
        VALUES (%s, %s, %s, %s)
    """, (username, product['Product Id'], product['Product Name'], product['Price']))
    conn.commit()
    conn.close()


# In[19]:


def load_cart_from_db(username):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT product_id FROM orders WHERE username = %s", (username,))
    product_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    st.session_state["cart"] = data[data['Product Id'].isin(product_ids)].to_dict('records')


# In[20]:


def add_to_favorites(username, product_id):
    try:
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT IGNORE INTO favorites (username, product_id) VALUES (%s, %s)", (username, product_id))
        conn.commit()
    except Exception as e:
        st.error(f"Error adding to favorites: {e}")
    finally:
        conn.close()


# In[21]:


def view_cart():
    st.title("🛒 Your Cart")
    if not st.session_state["cart"]:
        st.info("Your cart is empty.")
    else:
        total_items = len(st.session_state["cart"])
        st.markdown(f"### You have {total_items} item(s) in your cart:")
        for product in st.session_state["cart"]:
            with st.container():
                cols = st.columns([1, 3])
                with cols[0]:
                    st.image(product['Product Image Url'], width=100)
                with cols[1]:
                    st.markdown(f"**{product['Product Name']}**")
                    st.markdown(f"⭐ **Rating:** {product['Product Rating']}")
                    st.markdown(f"*{product['Product Description'][:150]}...*")
                    st.markdown(f"*{product['Product Description'][:150]}...*")
                with cols[2]:
                    if st.button("❌ Remove", key=f"remove_{product['Product Id']}"):
                        remove_from_cart(product['Product Id'])
                        st.experimental_rerun()
        st.markdown(f"### 🧾 Total: ${total_price:.2f}")
        if st.button("✅ Confirm Order"):
            st.success("🎉 Order confirmed! (Simulation)")
            st.session_state["cart"] = []


# In[22]:


# --- Streamlit UI ---
def login_page():
    st.title("🔐 Login to Product Recommender")

    option = st.radio("Choose an option:", ["Login", "Sign Up"])

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    if option == "Login":
        if st.button("Login"):
            if login_user(username, password):
                st.success(f"🎉 Welcome {username}!")
                st.rerun()

            else:
                st.error("❌ Invalid username or password.")
    
    elif option == "Sign Up":
        if st.button("Sign Up"):
            if signup_user(username, password):
                st.success("✅ Account created successfully! Please log in.")
            else:
                st.error("⚠️ Username already exists. Try a different one.")


# In[23]:


def home_page():
    st.markdown("<h1 style='text-align: center;'>🏠 Welcome to the Product Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>🛍️ Discover personalized products tailored to your taste!</p>", unsafe_allow_html=True)

    st.markdown("## 🔥 Trending Products")

    # Dummy trending product list if not already set
    if "trending" not in st.session_state:
        st.session_state["trending"] = [
            {
                
                "Product Id": 3,
                "Product Name": "Velvet Matte Lipstick",
                "Product Description": "A long-lasting, highly pigmented matte lipstick that glides on smoothly and stays put all day.",
                "Product Rating": 4.5,
                "Product Review": "The color is stunning and it doesn't dry out my lips. A must-have!",
                "Price": 19.99,
                "Product Image Url": "C:\\Users\\priya\\Downloads\\lipstick.png"

            },
            {
                "Product Id": 2,
                "Product Name": "Smart Watch",
                "Product Description": "...",
                "Product Rating": 4.6,
                "Product Review": "Tracks my fitness perfectly and looks great.",
                "Price": 79.99,
                "Product Image Url": "C:\\Users\\priya\\Downloads\\smart watch.png"
            },
            {
                "Product Id": 101,
                "Product Name": "Wireless Headphones",
                "Product Description": "High-quality over-ear headphones...",
                "Product Rating": 4.7,
                "Product Review": "These are the best headphones I've ever used!",
                "Price": 59.99,
                "Product Image Url":"C:\\Users\\priya\\Downloads\\headphone.png"
            }
        ]

    trending_products = st.session_state["trending"]

    if not trending_products:
        st.info("No trending products available right now.")
        return

    for product in trending_products:
        with st.container():
            cols = st.columns([1, 3])
            with cols[0]:
                try:
                    st.image(product["Product Image Url"], use_container_width=True)
                except:
                    st.image("https://via.placeholder.com/150?text=No+Image", use_container_width=True)
            with cols[1]:
                st.markdown(f"### **{product['Product Name']}**")
                st.markdown(f"⭐ **Rating:** {product['Product Rating']} &nbsp;&nbsp; 💰 **Price:** ${product['Price']:.2f}")
                st.markdown(f"📝 **Trending Review:** *{product['Product Review']}*")
        st.markdown("---")


# In[24]:


def recommendation_page():
    st.title("🛍️ Product Recommendation System")

    st.sidebar.write(f"👋 Welcome, {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        logout()
        st.rerun()
    
    st.sidebar.subheader("🌟 Top Rated Products")
    top_rated = data.sort_values(by="Product Rating", ascending=False).head(5)
    for _, row in top_rated.iterrows():
        st.sidebar.markdown(f"**{row['Product Name']}** — ⭐ {row['Product Rating']}")
    

    product_input = st.text_input("🔍 Enter a product name:")
    if "recent" not in st.session_state:
        st.session_state["recent"] = []
    
    min_rating = st.slider("Minimum Product Rating", min_value=1.0, max_value=5.0, step=0.5, value=3.0)
   
    if product_input and st.button("Get Recommendations"):
    # Track recently viewed
        st.session_state["recent"].append(product_input)
        st.session_state["recent"] = st.session_state["recent"][-5:]
        
        recommendations = recommend_products(product_input, min_rating=min_rating)

        if recommendations:
            st.subheader("🛒 Recommended Products")
            for product in recommendations:
                st.image(product['Product Image Url'], width=150)
                st.write(f"**{product['Product Name']}**")
                st.write(f"⭐ Rating: {product['Product Rating']}")
                # ❤️ Add to Favorites button
                if st.button(f"❤️ Add to Favorites", key=product['Product Id']):
                    add_to_favorites(st.session_state["username"], product['Product Id'])
                    st.success("Added to Favorites!")
        else:
            st.warning("⚠️ No recommendations found. Try a different product name.")
# Show recently viewed products
if "recent" in st.session_state and st.session_state["recent"]:
    st.subheader("🕓 Recently Viewed:")
    for p in st.session_state["recent"]:
        st.markdown(f"- {p}")


# In[25]:


def favorites_page():
    st.title("❤️ Your Favorite Products")

    username = st.session_state.get("username")
    if not username:
        st.warning("Please log in to view your favorites.")
        return

    try:
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT product_id FROM favorites WHERE username = %s", (username,))
        favorite_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        st.error(f"Error fetching favorites: {e}")
        return

    fav_products = data[data["Product Id"].isin(favorite_ids)]

    if not fav_products.empty:
        for _, row in fav_products.iterrows():
            with st.container():
                cols = st.columns([1, 3])
                with cols[0]:
                    if row["Product Image Url"]:
                        st.image(row["Product Image Url"], width=120)
                    else:
                        st.image("https://via.placeholder.com/120x150?text=No+Image", width=120)
                with cols[1]:
                    st.markdown(f"**{row['Product Name']}**")
                    st.markdown(f"⭐ **Rating:** {row['Product Rating']}")
                    st.markdown(f"*{row['Product Description'][:150]}...*")
    else:
        st.info("You haven’t added any favorites yet.")


# In[26]:


if not st.session_state["logged_in"]:
    login_page()
else:
    st.sidebar.title("📂 Navigate")
    page = st.sidebar.radio("Go to:", ["🏠 Home", "🛒 Recommend", "❤️ Favorites"])  # ✅ Add "❤️ Favorites" here

    if page == "🏠 Home":
        home_page()
    elif page == "🛒 Recommend":
        recommendation_page()
    elif page == "❤️ Favorites":          # ✅ Add this condition
        favorites_page()
    elif page == "🛍️ Cart":
        view_cart()


