from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import os
from bs4 import BeautifulSoup
import re
import json

class PartSelectScraper:
    """
    Scraper class using products array to scrape parts and repair pages,
    with dynamic URLs, checking robots.txt, and saving text files properly.
    """

    def __init__(self, driver_path=None):
        self.driver_path = driver_path
        self.products = ["Refrigerator", "Dishwasher"]  

    def check_robots_txt(self):
        robots_url = "https://www.partselect.com/robots.txt"
        disallowed_paths = []
        response = requests.get(robots_url)
        if response.status_code == 200:
            lines = response.text.splitlines()
            user_agent_all = False
            for line in lines:
                line = line.strip()
                if line.lower() == "user-agent: *":
                    user_agent_all = True
                elif user_agent_all and line.lower().startswith("disallow:"):
                    path = line.split(":", 1)[1].strip()
                    disallowed_paths.append(path)
                elif line.lower().startswith("user-agent:") and user_agent_all:
                    break
        return disallowed_paths

    def is_path_allowed(self, path, disallowed_paths):
        for disallowed in disallowed_paths:
            if disallowed == "":
                continue
            if disallowed.endswith("/") and path.startswith(disallowed):
                return False
            if path == disallowed:
                return False
        return True

    def _init_driver(self):
        if self.driver_path:
            return webdriver.Chrome(executable_path=self.driver_path)
        return webdriver.Chrome()
    
    def get_product_folder(self, product, save_dir, folder_type):
        product_lower = product.lower()
        folder_path = os.path.join(save_dir, product_lower, folder_type)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def _save_text_to_file(self, text, save_path, url=None):
        if save_path:
            dir_path = os.path.dirname(save_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                if url:
                    f.write(f"Source URL: {url}\n\n")
                f.write(text)

    def scrape_parts_page(self, product, save_dir="Scrapped Pages"):
        product_lower = product.lower()
        target_path = f"/{product}-Parts.htm"
        disallowed_paths = self.check_robots_txt()
        if not self.is_path_allowed(target_path, disallowed_paths):
            raise Exception(f"Access to path '{target_path}' is disallowed by robots.txt")
        
        driver = self._init_driver()
        try:
            url = f"https://www.partselect.com/{product}-Parts.htm"
            driver.get(url)
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            driver.execute_script("""
                const header = document.querySelector('header');
                if (header) header.remove();
                const footer = document.querySelector('footer');
                if (footer) footer.remove();
            """)

            page_text = driver.execute_script("""
                const main = document.querySelector('main') || document.body;
                return main.innerText;
            """)
            save_path = os.path.join(self.get_product_folder(product, save_dir, "parts"), f"{product_lower}_parts.txt")
            self._save_text_to_file(page_text, save_path, url=url)
            return page_text
        finally:
            driver.quit()

    def scrape_repair_page(self, product, save_dir="Scrapped Pages"):
        product_lower = product.lower()
        target_path = f"/Repair/{product}/"
        disallowed_paths = self.check_robots_txt()
        if not self.is_path_allowed(target_path, disallowed_paths):
            raise Exception(f"Access to path '{target_path}' is disallowed by robots.txt")

        driver = self._init_driver()
        try:
            url = f"https://www.partselect.com/Repair/{product}/"
            driver.get(url)
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            # Get the full page source after JavaScript loads content
            html = driver.page_source

            # Parse with BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            # Find all symptom anchor tags
            symptom_links = soup.find_all('a', href=True)
            symptom_data = []

            import urllib.parse

            for a in symptom_links:
                href = a['href']
                path_parts = urllib.parse.urlparse(href).path.strip("/").split("/")

                if (
                    len(path_parts) == 3 
                    and path_parts[0].lower() == "repair"
                    and path_parts[1].lower() == product.lower()
                ):
                    title_elem = a.find('h3')
                    title = title_elem.get_text(strip=True) if title_elem else a.get_text(strip=True)
                    if title:
                        full_url = f"https://www.partselect.com{href}"
                        symptom_data.append({
                            "title": title,
                            "url": full_url
                        })

            import json
            save_path = os.path.join(self.get_product_folder(product, save_dir, "repairs"), f"{product_lower}_repair_symptoms.json")
            os.makedirs(save_dir, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(symptom_data, f, ensure_ascii=False, indent=2)

            return symptom_data

        finally:
            driver.quit()
            
    @staticmethod
    def sanitize_filename(name):
        return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
    
    def process_symptoms(self, product, save_dir="Scrapped Pages"):
        product_lower = product.lower()
        symptoms_json_path = os.path.join(
            self.get_product_folder(product, save_dir, "symptoms"),
            f"{product_lower}_repair_symptoms.json"
        )

        if not os.path.exists(symptoms_json_path):
            print(f"Symptoms JSON file not found for {product}. Scraping repair page...")
            self.scrape_repair_page(product, save_dir=save_dir)

        with open(symptoms_json_path, "r", encoding="utf-8") as f:
            symptoms = json.load(f)
            driver = self._init_driver()
        try:
            for entry in symptoms:
                url = entry["url"]   
                title = entry["title"]
                filename = f"{product_lower}_{self.sanitize_filename(title)}_repair.txt"
                save_path = os.path.join(save_dir, filename)

                driver.get(url)
                wait = WebDriverWait(driver, 10)
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

                # Remove header and footer for clean content
                driver.execute_script("""
                    const header = document.querySelector('header');
                    if (header) header.remove();
                    const footer = document.querySelector('footer');
                    if (footer) footer.remove();
                """)
                # Grab main content, fallback to full body if 'main' not found
                page_text = driver.execute_script(
                    "return (document.querySelector('main') || document.body).innerText;"
                )

                os.makedirs(save_dir, exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as tf:
                    tf.write(f"Source URL: {url}\n\n")
                    tf.write(page_text)
        finally:
            driver.quit()

# scraper = PartSelectScraper()

# # for product in scraper.products:
#     # scraper.scrape_parts_page(product)
#     scraper.scrape_repair_page(product)
#     scraper.process_symptoms(product)
