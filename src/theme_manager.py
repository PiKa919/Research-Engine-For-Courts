"""
Dark Mode and Theme System for Legal Research Engine

This module provides theme switching capabilities and dark mode support
for the Streamlit interface.
"""

import streamlit as st
from typing import Literal, Dict, Any

ThemeType = Literal["light", "dark", "auto", "legal_blue", "legal_classic"]

class ThemeManager:
    """Manages application themes and styling"""
    
    def __init__(self):
        self.themes = {
            "light": self._light_theme(),
            "dark": self._dark_theme(),
            "auto": self._auto_theme(),
            "legal_blue": self._legal_blue_theme(),
            "legal_classic": self._legal_classic_theme()
        }
    
    def _light_theme(self) -> Dict[str, str]:
        return {
            "primary_bg": "#ffffff",
            "secondary_bg": "#f0f2f6",
            "text_color": "#262730",
            "accent_color": "#ff4b4b",
            "sidebar_bg": "#f0f2f6",
            "card_bg": "#ffffff",
            "border_color": "#e6e9f0",
            "success_color": "#00cc88",
            "warning_color": "#ffab00",
            "error_color": "#ff2b2b"
        }
    
    def _dark_theme(self) -> Dict[str, str]:
        return {
            "primary_bg": "#0e1117",
            "secondary_bg": "#1e1e1e",
            "text_color": "#fafafa",
            "accent_color": "#ff4b4b",
            "sidebar_bg": "#262730",
            "card_bg": "#1e1e1e",
            "border_color": "#3d3d3d",
            "success_color": "#00cc88",
            "warning_color": "#ffab00",
            "error_color": "#ff2b2b"
        }
    
    def _auto_theme(self) -> Dict[str, str]:
        """Auto theme based on system preference"""
        # For now, default to dark theme
        # In production, would detect system theme
        return self._dark_theme()
    
    def _legal_blue_theme(self) -> Dict[str, str]:
        """Professional legal blue theme"""
        return {
            "primary_bg": "#f8fafd",
            "secondary_bg": "#e8f4f8",
            "text_color": "#1a202c",
            "accent_color": "#3182ce",
            "sidebar_bg": "#e8f4f8",
            "card_bg": "#ffffff",
            "border_color": "#cbd5e0",
            "success_color": "#38a169",
            "warning_color": "#d69e2e",
            "error_color": "#e53e3e"
        }
    
    def _legal_classic_theme(self) -> Dict[str, str]:
        """Classic legal document theme"""
        return {
            "primary_bg": "#fffef7",
            "secondary_bg": "#f7f5f0",
            "text_color": "#2d3748",
            "accent_color": "#744c1e",
            "sidebar_bg": "#f7f5f0",
            "card_bg": "#ffffff",
            "border_color": "#d4cdbc",
            "success_color": "#38a169",
            "warning_color": "#d69e2e",
            "error_color": "#e53e3e"
        }
    
    def apply_theme(self, theme_name: ThemeType) -> None:
        """Apply the selected theme to the Streamlit app"""
        
        if theme_name not in self.themes:
            st.warning(f"Theme '{theme_name}' not found. Using light theme.")
            theme_name = "light"
        
        theme = self.themes[theme_name]
        
        # Store current theme in session state
        st.session_state.current_theme = theme_name
        
        # Apply CSS styling
        css = self._generate_theme_css(theme)
        st.markdown(css, unsafe_allow_html=True)
    
    def _generate_theme_css(self, theme: Dict[str, str]) -> str:
        """Generate CSS for the theme"""
        
        return f"""
        <style>
        /* Main app styling */
        .stApp {{
            background-color: {theme["primary_bg"]};
            color: {theme["text_color"]};
        }}
        
        /* Sidebar styling */
        .css-1d391kg {{
            background-color: {theme["sidebar_bg"]};
        }}
        
        /* Main content area */
        .main .block-container {{
            background-color: {theme["primary_bg"]};
            padding-top: 2rem;
        }}
        
        /* Cards and containers */
        .stExpander {{
            background-color: {theme["card_bg"]};
            border: 1px solid {theme["border_color"]};
            border-radius: 10px;
        }}
        
        /* Metrics */
        .metric-card {{
            background-color: {theme["card_bg"]};
            border: 1px solid {theme["border_color"]};
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        /* Chat messages */
        .stChatMessage {{
            background-color: {theme["card_bg"]};
            border: 1px solid {theme["border_color"]};
            border-radius: 10px;
        }}
        
        /* Buttons */
        .stButton > button {{
            background-color: {theme["accent_color"]};
            color: white;
            border: none;
            border-radius: 5px;
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            background-color: {theme["accent_color"]};
            opacity: 0.8;
            transform: translateY(-2px);
        }}
        
        /* Success elements */
        .stSuccess {{
            background-color: {theme["success_color"]};
            color: white;
        }}
        
        /* Warning elements */
        .stWarning {{
            background-color: {theme["warning_color"]};
            color: white;
        }}
        
        /* Error elements */
        .stError {{
            background-color: {theme["error_color"]};
            color: white;
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {theme["secondary_bg"]};
            border-radius: 10px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            color: {theme["text_color"]};
            border-radius: 5px;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {theme["accent_color"]};
            color: white;
        }}
        
        /* Data frames */
        .dataframe {{
            background-color: {theme["card_bg"]};
            color: {theme["text_color"]};
            border: 1px solid {theme["border_color"]};
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {theme["text_color"]};
        }}
        
        /* Code blocks */
        .stCodeBlock {{
            background-color: {theme["secondary_bg"]};
            border: 1px solid {theme["border_color"]};
            border-radius: 5px;
        }}
        
        /* Progress bars */
        .stProgress > div > div {{
            background-color: {theme["accent_color"]};
        }}
        
        /* Custom legal theme elements */
        .legal-header {{
            background: linear-gradient(135deg, {theme["accent_color"]} 0%, {theme["secondary_bg"]} 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
        }}
        
        .legal-card {{
            background: {theme["card_bg"]};
            border: 2px solid {theme["border_color"]};
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        
        .legal-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
        }}
        
        .status-indicator {{
            display: inline-flex;
            align-items: center;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0.2rem;
        }}
        
        .status-online {{
            background-color: {theme["success_color"]};
            color: white;
        }}
        
        .status-offline {{
            background-color: {theme["error_color"]};
            color: white;
        }}
        
        .status-warning {{
            background-color: {theme["warning_color"]};
            color: white;
        }}
        
        /* Animation effects */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .fade-in {{
            animation: fadeIn 0.6s ease-out;
        }}
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {{
            width: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {theme["secondary_bg"]};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {theme["border_color"]};
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {theme["accent_color"]};
        }}
        </style>
        """

def create_theme_selector() -> None:
    """Create theme selection interface in sidebar"""
    
    st.sidebar.markdown("### 🎨 Theme Settings")
    
    # Get current theme
    current_theme = st.session_state.get("current_theme", "light")
    
    # Theme options with descriptions
    theme_options = {
        "light": "☀️ Light Mode",
        "dark": "🌙 Dark Mode", 
        "auto": "🔄 Auto (System)",
        "legal_blue": "⚖️ Legal Blue",
        "legal_classic": "📜 Legal Classic"
    }
    
    # Theme selector
    selected_theme = st.sidebar.selectbox(
        "Choose Theme",
        options=list(theme_options.keys()),
        format_func=lambda x: theme_options[x],
        index=list(theme_options.keys()).index(current_theme)
    )
    
    # Apply theme if changed
    if selected_theme != current_theme:
        theme_manager = ThemeManager()
        theme_manager.apply_theme(selected_theme)
        st.rerun()
    
    # Theme preview
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Theme Preview:**")
    
    # Show current theme colors
    if "current_theme" in st.session_state:
        theme_manager = ThemeManager()
        current_theme_colors = theme_manager.themes[st.session_state.current_theme]
        
        st.sidebar.markdown(
            f"""
            <div style="
                background: {current_theme_colors['card_bg']};
                border: 2px solid {current_theme_colors['accent_color']};
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
                color: {current_theme_colors['text_color']};
            ">
                <strong>Sample Card</strong><br>
                This shows how content looks with the current theme.
            </div>
            """,
            unsafe_allow_html=True
        )

def apply_theme_on_startup():
    """Apply saved theme on app startup"""
    
    # Get saved theme from session state or default to light
    saved_theme = st.session_state.get("current_theme", "light")
    
    # Initialize theme manager and apply theme
    theme_manager = ThemeManager()
    theme_manager.apply_theme(saved_theme)

def create_legal_styled_header(title: str, subtitle: str = "") -> None:
    """Create a professionally styled header for legal applications"""
    
    st.markdown(
        f"""
        <div class="legal-header fade-in">
            <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">
                {title}
            </h1>
            {f'<p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">{subtitle}</p>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

def create_status_badge(text: str, status: Literal["online", "offline", "warning"]) -> str:
    """Create a status badge with appropriate styling"""
    
    icons = {
        "online": "🟢",
        "offline": "🔴", 
        "warning": "🟡"
    }
    
    return f"""
    <span class="status-indicator status-{status}">
        {icons[status]} {text}
    </span>
    """

# Export main functions
__all__ = [
    'ThemeManager',
    'create_theme_selector',
    'apply_theme_on_startup',
    'create_legal_styled_header',
    'create_status_badge'
]