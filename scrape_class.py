from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests


class PartSelectScraper:
    """
    A scraper class for interacting with partselect.com to:
    - Check robots.txt and validate allowed paths
    - Scrape refrigerator parts page
    - Scrape dishwasher parts page
    - Search for a specific part number
    """

    def __init__(self, driver_path=None):
        """
        Initialize the scraper. Optionally specify the path to chromedriver if not in PATH.
        """
        self.driver_path = driver_path

    def check_robots_txt(self):
        """
        Fetches the robots.txt file from partselect.com and returns disallowed paths for user-agent '*'.
        """
        robots_url = "https://www.partselect.com/robots.txt"
        disallowed_paths = []
        response = requests.get(robots_url)
        if response.status_code == 200:
            lines = response.text.splitlines()
            user_agent_all = False
            for line in lines:
                line = line.strip()
                if line.lower().startswith("user-agent: *"):
                    user_agent_all = True
                elif user_agent_all and line.lower().startswith("disallow:"):
                    path = line.split(":", 1)[1].strip()
                    disallowed_paths.append(path)
                elif line.lower().startswith("user-agent:") and user_agent_all:
                    # Stop reading disallow rules once another user-agent block starts
                    break
        return disallowed_paths

    def is_path_allowed(self, path, disallowed_paths):
        """
        Checks if the given path is allowed to be accessed based on the list of disallowed paths.
        """
        for disallowed in disallowed_paths:
            if disallowed == "":
                continue
            if disallowed.endswith("/") and path.startswith(disallowed):
                return False
            if path == disallowed:
                return False
        return True

    def _init_driver(self):
        """
        Helper method to initialize the Chrome WebDriver.
        """
        if self.driver_path:
            return webdriver.Chrome(executable_path=self.driver_path)
        else:
            return webdriver.Chrome()

    def scrape_refrigerator_parts_page(self):
        """
        Scrapes the refrigerator parts page content if allowed by robots.txt.
        Returns the visible main content text.
        """
        disallowed_paths = self.check_robots_txt()
        target_path = "/Refrigerator-Parts.htm"
        if not self.is_path_allowed(target_path, disallowed_paths):
            raise Exception(f"Access to path '{target_path}' is disallowed by robots.txt")

        driver = self._init_driver()
        try:
            url = "https://www.partselect.com/Refrigerator-Parts.htm"
            driver.get(url)
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            # Remove header and footer to reduce noise
            driver.execute_script("""
                const header = document.querySelector('header');
                if (header) { header.remove(); }
                const footer = document.querySelector('footer');
                if (footer) { footer.remove(); }
            """)

            # Return main content text
            page_text = driver.execute_script("""
                const main = document.querySelector('main') || document.body;
                return main.innerText;
            """)
            return page_text
        finally:
            driver.quit()

    def scrape_dishwasher_parts_page(self):
        """
        Scrapes the dishwasher parts page content if allowed by robots.txt.
        Returns the visible main content text.
        """
        disallowed_paths = self.check_robots_txt()
        target_path = "/Dishwasher-Parts.htm"
        if not self.is_path_allowed(target_path, disallowed_paths):
            raise Exception(f"Access to path '{target_path}' is disallowed by robots.txt")

        driver = self._init_driver()
        try:
            url = "https://www.partselect.com/Dishwasher-Parts.htm"
            driver.get(url)
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            driver.execute_script("""
                const header = document.querySelector('header');
                if (header) { header.remove(); }
                const footer = document.querySelector('footer');
                if (footer) { footer.remove(); }
            """)

            page_text = driver.execute_script("""
                const main = document.querySelector('main') || document.body;
                return main.innerText;
            """)
            return page_text
        finally:
            driver.quit()

    def search_part(self, part_number):
        """
        Searches for a specific part number on the homepage search bar.
        Returns visible page text after search results load.
        """
        disallowed_paths = self.check_robots_txt()
        search_path = "/"
        if not self.is_path_allowed(search_path, disallowed_paths):
            raise Exception(f"Access to path '{search_path}' is disallowed by robots.txt")

        driver = self._init_driver()
        try:
            driver.get("https://www.partselect.com/")

            wait = WebDriverWait(driver, 10)
            search_input = wait.until(EC.presence_of_element_located((By.ID, "searchboxInput")))
            search_input.clear()
            search_input.send_keys(part_number)

            search_button = driver.find_element(By.CLASS_NAME, "js-searchBtn")
            search_button.click()

            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            page_text = driver.execute_script("return document.documentElement.innerText;")
            return page_text
        finally:
            driver.quit()