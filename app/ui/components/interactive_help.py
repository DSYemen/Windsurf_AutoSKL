import streamlit as st
from typing import Dict, List, Optional
import json
from pathlib import Path
import markdown2
import re

class InteractiveHelp:
    def __init__(self, help_file: str):
        """Initialize the interactive help system"""
        self.help_content = self._load_help_content(help_file)
        self.search_index = self._create_search_index()
        
    def _load_help_content(self, help_file: str) -> str:
        """Load help content from markdown file"""
        return Path(help_file).read_text(encoding='utf-8')
        
    def _create_search_index(self) -> Dict[str, List[str]]:
        """Create a simple search index from help content"""
        index = {}
        
        # Split content into sections
        sections = re.split(r'^##\s+', self.help_content, flags=re.MULTILINE)
        
        for section in sections:
            if not section.strip():
                continue
                
            # Get section title and content
            lines = section.split('\n')
            title = lines[0].strip()
            content = '\n'.join(lines[1:]).strip()
            
            # Add to index
            words = set(re.findall(r'\w+', section.lower()))
            for word in words:
                if word not in index:
                    index[word] = []
                index[word].append(title)
                
        return index
        
    def show_help_sidebar(self):
        """Show help sidebar with search and navigation"""
        with st.sidebar:
            st.markdown("## 🔍 Help & Documentation")
            
            # Search
            search_query = st.text_input(
                "Search help...",
                placeholder="Type to search..."
            )
            
            if search_query:
                self.show_search_results(search_query)
                
            # Quick links
            st.markdown("### Quick Links")
            if st.button("📊 Data Analysis"):
                self.show_section("Data Analysis")
            if st.button("🔧 Model Training"):
                self.show_section("Model Training")
            if st.button("📈 Model Evaluation"):
                self.show_section("Model Evaluation")
            if st.button("🎯 Making Predictions"):
                self.show_section("Making Predictions")
                
            # Show all sections
            with st.expander("📚 All Topics", expanded=False):
                sections = re.findall(
                    r'^##\s+(.+)$',
                    self.help_content,
                    flags=re.MULTILINE
                )
                for section in sections:
                    if st.button(section, key=f"section_{section}"):
                        self.show_section(section)
                        
    def show_search_results(self, query: str):
        """Show search results for query"""
        # Tokenize query
        query_words = set(re.findall(r'\w+', query.lower()))
        
        # Find matching sections
        matching_sections = set()
        for word in query_words:
            if word in self.search_index:
                matching_sections.update(self.search_index[word])
                
        if matching_sections:
            st.markdown("### Search Results")
            for section in matching_sections:
                if st.button(f"📍 {section}", key=f"search_{section}"):
                    self.show_section(section)
        else:
            st.info("No matching topics found")
            
    def show_section(self, section_title: str):
        """Show a specific help section"""
        # Find section content
        pattern = f"## {section_title}.*?(?=##|$)"
        match = re.search(
            pattern,
            self.help_content,
            flags=re.DOTALL | re.MULTILINE
        )
        
        if match:
            content = match.group(0)
            # Convert markdown to HTML
            html = markdown2.markdown(
                content,
                extras=['fenced-code-blocks', 'tables']
            )
            st.markdown(html, unsafe_allow_html=True)
            
            # Related topics
            self._show_related_topics(section_title)
        else:
            st.error(f"Section '{section_title}' not found")
            
    def _show_related_topics(self, current_section: str):
        """Show related help topics"""
        # Simple related topics based on common words
        section_words = set(re.findall(r'\w+', current_section.lower()))
        
        related = []
        sections = re.findall(
            r'^##\s+(.+)$',
            self.help_content,
            flags=re.MULTILINE
        )
        
        for section in sections:
            if section == current_section:
                continue
            words = set(re.findall(r'\w+', section.lower()))
            common = len(section_words & words)
            if common > 0:
                related.append((section, common))
                
        if related:
            st.markdown("### Related Topics")
            related.sort(key=lambda x: x[1], reverse=True)
            for section, _ in related[:3]:
                if st.button(f"👉 {section}", key=f"related_{section}"):
                    self.show_section(section)
                    
    def show_interactive_guide(self):
        """Show interactive guide with steps"""
        st.markdown("## 🚀 Interactive Guide")
        
        # Task selection
        task = st.selectbox(
            "What would you like to do?",
            [
                "Analyze Data",
                "Train a Model",
                "Evaluate Model Performance",
                "Make Predictions",
                "Troubleshoot Issues"
            ]
        )
        
        if task == "Analyze Data":
            self._show_data_analysis_guide()
        elif task == "Train a Model":
            self._show_model_training_guide()
        elif task == "Evaluate Model Performance":
            self._show_evaluation_guide()
        elif task == "Make Predictions":
            self._show_prediction_guide()
        elif task == "Troubleshoot Issues":
            self._show_troubleshooting_guide()
            
    def _show_data_analysis_guide(self):
        """Show interactive guide for data analysis"""
        steps = [
            "Upload your dataset (CSV format)",
            "Select the target column",
            "Review dataset overview",
            "Analyze feature distributions",
            "Check correlations",
            "Review preprocessing recommendations"
        ]
        
        current_step = st.session_state.get('guide_step', 0)
        
        # Progress bar
        st.progress(current_step / len(steps))
        
        # Current step
        st.markdown(f"### Step {current_step + 1}: {steps[current_step]}")
        
        # Step-specific content
        if current_step == 0:
            st.markdown("""
                📤 **Upload your data:**
                - Use CSV format
                - Ensure clean column names
                - Check for missing values
            """)
        elif current_step == 1:
            st.markdown("""
                🎯 **Select target:**
                - Choose the column to predict
                - Ensure correct data type
                - Check class distribution
            """)
        # ... Add content for other steps
        
        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if current_step > 0:
                if st.button("⬅️ Previous"):
                    st.session_state.guide_step = current_step - 1
                    st.experimental_rerun()
                    
        with col2:
            if current_step < len(steps) - 1:
                if st.button("Next ➡️"):
                    st.session_state.guide_step = current_step + 1
                    st.experimental_rerun()
                    
    def _show_model_training_guide(self):
        """Show interactive guide for model training"""
        # Similar to _show_data_analysis_guide
        pass
        
    def _show_evaluation_guide(self):
        """Show interactive guide for model evaluation"""
        # Similar to _show_data_analysis_guide
        pass
        
    def _show_prediction_guide(self):
        """Show interactive guide for making predictions"""
        # Similar to _show_data_analysis_guide
        pass
        
    def _show_troubleshooting_guide(self):
        """Show interactive guide for troubleshooting"""
        # Similar to _show_data_analysis_guide
        pass
