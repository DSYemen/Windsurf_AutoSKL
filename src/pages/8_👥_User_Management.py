import streamlit as st
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from src.utils.rtl_utils import apply_arabic_config

# تطبيق التكوين العربي
apply_arabic_config(title="إدارة المستخدمين", icon="👥")

# Constants
USERS_FILE = Path("data/users.json")
SECRET_KEY = "your-secret-key"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# User roles and permissions
ROLES = {
    "admin": {
        "can_manage_users": True,
        "can_train_models": True,
        "can_deploy_models": True,
        "can_view_all_models": True
    },
    "data_scientist": {
        "can_manage_users": False,
        "can_train_models": True,
        "can_deploy_models": True,
        "can_view_all_models": True
    },
    "viewer": {
        "can_manage_users": False,
        "can_train_models": False,
        "can_deploy_models": False,
        "can_view_all_models": True
    }
}

def init_users_file():
    """Initialize users file if it doesn't exist"""
    if not USERS_FILE.exists():
        USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        default_admin = {
            "username": "admin",
            "password": hash_password("admin"),  # Change in production
            "role": "admin",
            "created_at": datetime.now().isoformat(),
            "last_login": None
        }
        save_users({"admin": default_admin})

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users() -> dict:
    """Load users from JSON file"""
    if USERS_FILE.exists():
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def save_users(users: dict):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.JWTError:
        return None

def check_permission(permission: str) -> bool:
    """Check if current user has permission"""
    if "user_token" not in st.session_state:
        return False
    
    payload = verify_token(st.session_state.user_token)
    if not payload:
        return False
    
    role = payload.get("role")
    return ROLES.get(role, {}).get(permission, False)

# Initialize users file
init_users_file()

# Main layout
st.title("👥 User Management")

# Sidebar for login/logout
with st.sidebar:
    if "user_token" not in st.session_state:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            users = load_users()
            user = users.get(username)
            
            if user and user["password"] == hash_password(password):
                # Create access token
                token = create_access_token({
                    "sub": username,
                    "role": user["role"]
                })
                
                # Update last login
                user["last_login"] = datetime.now().isoformat()
                save_users(users)
                
                # Store token
                st.session_state.user_token = token
                st.success(f"Welcome {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    else:
        if st.button("Logout"):
            del st.session_state.user_token
            st.rerun()

# Main content
if "user_token" in st.session_state:
    payload = verify_token(st.session_state.user_token)
    if not payload:
        del st.session_state.user_token
        st.error("Session expired. Please login again.")
        st.rerun()
    
    current_user = payload["sub"]
    current_role = payload["role"]
    
    st.info(f"Logged in as: {current_user} (Role: {current_role})")
    
    if check_permission("can_manage_users"):
        # User management section
        st.subheader("Manage Users")
        
        # Add new user
        with st.expander("Add New User"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", list(ROLES.keys()))
            
            if st.button("Add User"):
                users = load_users()
                if new_username in users:
                    st.error("Username already exists")
                else:
                    users[new_username] = {
                        "username": new_username,
                        "password": hash_password(new_password),
                        "role": new_role,
                        "created_at": datetime.now().isoformat(),
                        "last_login": None
                    }
                    save_users(users)
                    st.success(f"User {new_username} added successfully")
        
        # List and manage users
        st.subheader("User List")
        users = load_users()
        user_list = []
        
        for username, user in users.items():
            user_list.append({
                "Username": username,
                "Role": user["role"],
                "Created": datetime.fromisoformat(user["created_at"]).strftime("%Y-%m-%d %H:%M"),
                "Last Login": datetime.fromisoformat(user["last_login"]).strftime("%Y-%m-%d %H:%M") if user["last_login"] else "Never"
            })
        
        df = pd.DataFrame(user_list)
        st.dataframe(df, use_container_width=True)
        
        # Delete user
        with st.expander("Delete User"):
            user_to_delete = st.selectbox(
                "Select User to Delete",
                [u for u in users.keys() if u != current_user]
            )
            
            if st.button("Delete User"):
                if user_to_delete:
                    users = load_users()
                    del users[user_to_delete]
                    save_users(users)
                    st.success(f"User {user_to_delete} deleted successfully")
                    st.rerun()
        
        # Change user role
        with st.expander("Change User Role"):
            user_to_change = st.selectbox(
                "Select User",
                [u for u in users.keys() if u != current_user],
                key="change_role"
            )
            new_role = st.selectbox(
                "New Role",
                list(ROLES.keys()),
                key="new_role"
            )
            
            if st.button("Change Role"):
                if user_to_change:
                    users = load_users()
                    users[user_to_change]["role"] = new_role
                    save_users(users)
                    st.success(f"Role updated for {user_to_change}")
                    st.rerun()
    
    else:
        st.warning("You don't have permission to manage users")
        
    # User profile section
    st.subheader("My Profile")
    users = load_users()
    user = users[current_user]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"Username: {user['username']}")
        st.write(f"Role: {user['role']}")
        st.write(f"Account Created: {datetime.fromisoformat(user['created_at']).strftime('%Y-%m-%d %H:%M')}")
        if user['last_login']:
            st.write(f"Last Login: {datetime.fromisoformat(user['last_login']).strftime('%Y-%m-%d %H:%M')}")
    
    with col2:
        # Change password
        with st.expander("Change Password"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            if st.button("Update Password"):
                if hash_password(current_password) != user["password"]:
                    st.error("Current password is incorrect")
                elif new_password != confirm_password:
                    st.error("New passwords don't match")
                else:
                    users[current_user]["password"] = hash_password(new_password)
                    save_users(users)
                    st.success("Password updated successfully")
else:
    st.warning("Please login to access this page")
