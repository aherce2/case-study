# scrape.py
import os
import requests
import json
import re
import urllib.parse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
from bs4 import BeautifulSoup

class PartSelectScraper:
    def __init__(self, driver_path=None, products=None):
        self.driver_path = driver_path
        self.products = products or ["Refrigerator", "Dishwasher"]
    
    def _init_driver(self):
        if self.driver_path:
            return webdriver.Chrome(executable_path=self.driver_path)
        return webdriver.Chrome()

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

    def close_interrupting_popups(self, driver):
        try:
            pop_up_close_btn = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '.modal-close, .popup-close, .close-button, .overlay-close'))
            )
            pop_up_close_btn.click()
            WebDriverWait(driver, 2).until(EC.invisibility_of_element(pop_up_close_btn))
        except TimeoutException:
            pass

    def get_product_folder(self, product, base_dir, folder_type):
        product_lower = product.lower()
        path = os.path.join(base_dir, product_lower, folder_type)
        os.makedirs(path, exist_ok=True)
        return path

    def _save_text(self, text, filename, url=None):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            if url:
                f.write(f"Source URL: {url}\n\n")
            f.write(text)

    def scrape_parts_page(self, product, save_dir="Scrapped Pages"):
        product_lower = product.lower()
        target_path = f"/{product}-Parts.htm"
        if not self.is_path_allowed(target_path, self.check_robots_txt()):
            raise Exception(f"Access disallowed by robots.txt: {target_path}")

        driver = self._init_driver()
        try:
            url = f"https://www.partselect.com/{product}-Parts.htm"
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            # Remove header/footer for cleaner text
            driver.execute_script("document.querySelector('header')?.remove(); document.querySelector('footer')?.remove();")
            page_text = driver.execute_script("(document.querySelector('main') || document.body).innerText")
            save_path = os.path.join(self.get_product_folder(product, save_dir, "parts"), f"{product_lower}_parts.txt")
            self._save_text(page_text, save_path, url)
            return page_text
        finally:
            driver.quit()

    # Additional scraping methods follow similar safe pattern with error handling,
    # popup closing, and saving results (repair pages, model pages, symptom processing, etc.)

    def get_part_information(self, model_number):
        search_path = "/"
        if not self.is_path_allowed(search_path, self.check_robots_txt()):
            raise Exception(f"Access disallowed by robots.txt for path {search_path}")

        driver = self._init_driver()
        try:
            url = "https://www.partselect.com/"
            driver.get(url)
            wait = WebDriverWait(driver, 10)
            search_input = wait.until(EC.presence_of_element_located((By.ID, "searchboxInput")))
            search_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.js-searchBtn")))

            search_input.clear()
            search_input.send_keys(model_number)

            self.close_interrupting_popups(driver)
            driver.execute_script("arguments[0].scrollIntoView(true);", search_button)

            try:
                search_button.click()
            except ElementClickInterceptedException:
                self.close_interrupting_popups(driver)
                driver.execute_script("arguments[0].click();", search_button)

            wait.until(EC.url_changes(url))
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            driver.execute_script("document.querySelector('header')?.remove(); document.querySelector('footer')?.remove();")

            def safe_find_text(by, locator):
                try:
                    return driver.find_element(by, locator).text.strip()
                except:
                    return None

            def safe_find_attr(by, locator, attr):
                try:
                    return driver.find_element(by, locator).get_attribute(attr)
                except:
                    return None

            part_info = {
                'partselect_number': safe_find_text(By.CSS_SELECTOR, '[itemprop="productID"]'),
                'manufacturer_part_number': safe_find_text(By.CSS_SELECTOR, '[itemprop="mpn"]'),
                'title': safe_find_text(By.CSS_SELECTOR, 'h1[itemprop="name"]'),
                'price': safe_find_text(By.CSS_SELECTOR, 'span.price[itemprop="price"]'),
                'availability': safe_find_text(By.CSS_SELECTOR, '[itemprop="availability"]'),
                'description': safe_find_text(By.CSS_SELECTOR, '[itemprop="description"]'),
                'reviews': safe_find_text(By.CSS_SELECTOR, '.rating__count'),
                'main_image_url': safe_find_attr(By.CSS_SELECTOR, '.main-media.MagicZoom-PartImage img', 'src'),
            }

            try:
                brand_elem = driver.find_element(By.CSS_SELECTOR, '[itemprop="brand"] [itemprop="name"]')
                part_info['brand'] = brand_elem.text.strip()
                brand_parent = brand_elem.find_element(By.XPATH, "..")
                compat_elem = brand_parent.find_element(By.XPATH, "following-sibling::span[1]")
                part_info['compatibility'] = compat_elem.text.strip() if compat_elem else None
            except:
                part_info['brand'] = None
                part_info['compatibility'] = None

            return {'url': driver.current_url, 'part_info': part_info}
        finally:
            driver.quit()
