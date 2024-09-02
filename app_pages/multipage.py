import streamlit as st
import matplotlib.pyplot as plt


# Class to generate multiple Streamlit pages using an object oriented approach
class MultiPage:

    def __init__(self, app_name) -> None:
        # Initialize an empty list to store pages
        self.pages = []
        self.app_name = app_name

        # Set the configuration for the Streamlit app page
        st.set_page_config(
            page_title=self.app_name,
            page_icon="🍃")

    # Add a new page to the app
    def add_page(self, title, func) -> None:
        self.pages.append({"title": title, "function": func})

    # Method to run the app and display the selected page
    def run(self):
        # Set the main title of the app to the app name
        st.title(self.app_name)
        # Set sidebar radio button menu to select the page & displays the titles of the pages and returns the selected page
        page = st.sidebar.radio('Menu', self.pages, format_func=lambda page: page['title'])
        # Call the function of the selected page to display its content
        page['function']()