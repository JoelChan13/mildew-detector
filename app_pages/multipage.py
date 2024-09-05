import streamlit as st
import matplotlib.pyplot as plt


class MultiPage:
    """
    A class to generate Streamlit pages using an object-oriented approach.

    Attributes:
    pages (list): A list to store the pages of the app.
    app_name (str): The name of the Streamlit app.
    """

    def __init__(self, app_name) -> None:
        """
        Initializes MultiPage object with app & and sets page configuration.

        Parameters:
        app_name (str): The name of the Streamlit app.
        """
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon="🍃")

    def add_page(self, title, func) -> None:
        """
        Adds a new page to the app.

        Parameters:
        title (str): The title of the page.
        func (function): The function that renders the content of the page.
        """
        self.pages.append({"title": title, "function": func})

    def run(self):
        """
        Runs the app and displays the selected page.

        The method displays the app title,
        shows a sidebar with the available pages,
        and calls the function of the selected page to display its content.
        """
        st.title(self.app_name)
        page = st.sidebar.radio(
            'Menu', self.pages,
            format_func=lambda page: page['title'])
        page['function']()
