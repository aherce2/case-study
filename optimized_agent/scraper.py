import os
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException, WebDriverException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PartInfo:
    """Structured data class for part information"""
    partselect_number: Optional[str] = None
    manufacturer_part_number: Optional[str] = None
    title: Optional[str] = None
    price: Optional[str] = None
    availability: Optional[str] = None
    description: Optional[str] = None
    reviews: Optional[str] = None
    main_image_url: Optional[str] = None
    brand: Optional[str] = None
    compatibility: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ScrapingResult:
    """Structured result from scraping operations"""
    url: str
    part_info: PartInfo
    success: bool = True
    error_message: Optional[str] = None

class PartSelectScraper:
    
    BASE_URL = "https://www.partselect.com"
    ROBOTS_URL = f"{BASE_URL}/robots.txt"
    DEFAULT_TIMEOUT = 10
    
    # Common selectors consolidated
    SELECTORS = {
        'search_input': '#searchboxInput',
        'search_button': 'button.js-searchBtn',
        'popup_decline': 'button.bx-button[type="reset"][aria-label="Decline; close the dialog"]',
        'popup_slab': '.bx-slab',
        'part_search_input': 'input[aria-label="Enter a part description"]',
        'part_search_button': 'button.search-btn[aria-label="search"]',
        'part_info': {
            'partselect_number': '[itemprop="productID"]',
            'manufacturer_part_number': '[itemprop="mpn"]', 
            'title': 'h1[itemprop="name"]',
            'price': 'span.price[itemprop="price"]',
            'availability': '[itemprop="availability"]',
            'description': '[itemprop="description"]',
            'reviews': '.rating__count',
            'main_image_url': '.main-media.MagicZoom-PartImage img',
            'brand': '[itemprop="brand"] [itemprop="name"]'
        }
    }
    
    def __init__(self, driver_path: Optional[str] = None, 
                 products: Optional[List[str]] = None,
                 timeout: int = DEFAULT_TIMEOUT,
                 save_dir: str = "part_select_data"):
        self.driver_path = driver_path
        self.products = products or ["Refrigerator", "Dishwasher"]
        self.timeout = timeout
        self.save_dir = save_dir
        self._disallowed_paths = None
        
    @contextmanager
    def get_driver(self):
        """Context manager for WebDriver to ensure cleanup"""
        driver = None
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            
            if self.driver_path:
                driver = webdriver.Chrome(executable_path=self.driver_path, options=options)
            else:
                driver = webdriver.Chrome(options=options)
            yield driver
        except Exception as e:
            logger.error(f"Driver initialization failed: {e}")
            raise
        finally:
            if driver:
                driver.quit()

    def _get_disallowed_paths(self) -> List[str]:
        """Cache and return robots.txt disallowed paths"""
        if self._disallowed_paths is not None:
            return self._disallowed_paths
            
        self._disallowed_paths = []
        try:
            response = requests.get(self.ROBOTS_URL, timeout=5)
            if response.status_code == 200:
                lines = response.text.splitlines()
                user_agent_all = False
                for line in lines:
                    line = line.strip()
                    if line.lower() == "user-agent: *":
                        user_agent_all = True
                    elif user_agent_all and line.lower().startswith("disallow:"):
                        path = line.split(":", 1)[1].strip()
                        self._disallowed_paths.append(path)
                    elif line.lower().startswith("user-agent:") and user_agent_all:
                        break
        except requests.RequestException as e:
            logger.warning(f"Could not fetch robots.txt: {e}")
        
        return self._disallowed_paths

    def _is_path_allowed(self, path: str) -> bool:
        """Check if path is allowed by robots.txt"""
        disallowed_paths = self._get_disallowed_paths()
        for disallowed in disallowed_paths:
            if not disallowed:
                continue
            if disallowed.endswith("/") and path.startswith(disallowed):
                return False
            if path == disallowed:
                return False
        return True

    def _handle_popups(self, driver: webdriver.Chrome) -> None:
        """Handle interrupting popups - consolidated logic"""
        try:
            WebDriverWait(driver, 2).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, self.SELECTORS['popup_slab']))
            )
            decline_btn = WebDriverWait(driver, 2).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, self.SELECTORS['popup_decline']))
            )
            decline_btn.click()
            logger.info("Popup closed successfully")
        except (TimeoutException, WebDriverException):
            pass  # No popup present

    def _clean_page(self, driver: webdriver.Chrome) -> None:
        """Remove header/footer for cleaner content"""
        driver.execute_script("""
            document.querySelector('header')?.remove();
            document.querySelector('footer')?.remove();
        """)

    def _safe_find_element(self, driver: webdriver.Chrome, selector: str, 
                          attribute: Optional[str] = None) -> Optional[str]:
        """Safely find element text or attribute"""
        try:
            element = driver.find_element(By.CSS_SELECTOR, selector)
            if attribute:
                return element.get_attribute(attribute)
            return element.text.strip()
        except Exception:
            return None

    def _extract_part_info(self, driver: webdriver.Chrome) -> PartInfo:
        """Extract part information using consolidated selectors"""
        selectors = self.SELECTORS['part_info']
        
        part_info = PartInfo(
            partselect_number=self._safe_find_element(driver, selectors['partselect_number']),
            manufacturer_part_number=self._safe_find_element(driver, selectors['manufacturer_part_number']),
            title=self._safe_find_element(driver, selectors['title']),
            price=self._safe_find_element(driver, selectors['price']),
            availability=self._safe_find_element(driver, selectors['availability']),
            description=self._safe_find_element(driver, selectors['description']),
            reviews=self._safe_find_element(driver, selectors['reviews']),
            main_image_url=self._safe_find_element(driver, selectors['main_image_url'], 'src'),
            brand=self._safe_find_element(driver, selectors['brand'])
        )
        
        if part_info.brand:
            try:
                brand_elem = driver.find_element(By.CSS_SELECTOR, selectors['brand'])
                brand_parent = brand_elem.find_element(By.XPATH, "..")
                compat_elem = brand_parent.find_element(By.XPATH, "following-sibling::span[1]")
                part_info.compatibility = compat_elem.text.strip() if compat_elem else None
            except Exception:
                part_info.compatibility = None
                
        return part_info

    def _perform_search(self, driver: webdriver.Chrome, search_term: str) -> None:
        """Perform search with error handling"""
        wait = WebDriverWait(driver, self.timeout)
        
        search_input = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, self.SELECTORS['search_input']))
        )
        search_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, self.SELECTORS['search_button']))
        )
        
        search_input.clear()
        search_input.send_keys(search_term)
        
        self._handle_popups(driver)
        driver.execute_script("arguments[0].scrollIntoView(true);", search_button)
        
        try:
            search_button.click()
        except ElementClickInterceptedException:
            self._handle_popups(driver)
            driver.execute_script("arguments[0].click();", search_button)

    def _save_content(self, content: str, filename: str, url: Optional[str] = None) -> None:
        """Save content to file with URL header"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            if url:
                f.write(f"Source URL: {url}\n\n")
            f.write(content)

    def _get_save_path(self, product: str, folder_type: str, filename: str) -> str:
        """Generate save path for files"""
        product_lower = product.lower()
        path = os.path.join(self.save_dir, product_lower, folder_type)
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, filename)

    def scrape_parts_page(self, product: str) -> str:
        """Scrape product parts page"""
        target_path = f"/{product}-Parts.htm"
        if not self._is_path_allowed(target_path):
            raise PermissionError(f"Access disallowed by robots.txt: {target_path}")

        with self.get_driver() as driver:
            url = f"{self.BASE_URL}/{product}-Parts.htm"
            driver.get(url)
            
            WebDriverWait(driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            self._clean_page(driver)
            page_text = driver.execute_script(
                "return (document.querySelector('main') || document.body).innerText"
            )
            
            save_path = self._get_save_path(product, "parts", f"{product.lower()}_parts.txt")
            self._save_content(page_text, save_path, url)
            
            return page_text

    def get_part_information(self, model_number: str) -> ScrapingResult:
        """Get part information by model number"""
        if not self._is_path_allowed("/"):
            raise PermissionError("Access disallowed by robots.txt for path /")

        try:
            with self.get_driver() as driver:
                driver.get(self.BASE_URL)
                self._perform_search(driver, model_number)
                
                WebDriverWait(driver, self.timeout).until(EC.url_changes(self.BASE_URL))
                WebDriverWait(driver, self.timeout).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                self._clean_page(driver)
                part_info = self._extract_part_info(driver)
                
                return ScrapingResult(
                    url=driver.current_url,
                    part_info=part_info,
                    success=True
                )
                
        except Exception as e:
            logger.error(f"Error getting part information for {model_number}: {e}")
            return ScrapingResult(
                url="",
                part_info=PartInfo(),
                success=False,
                error_message=str(e)
            )

    def check_model_part_compatibility(self, model_number: str, part_number: str) -> ScrapingResult:
        """Check compatibility between model and part"""
        if not self._is_path_allowed("/"):
            raise PermissionError("Access disallowed by robots.txt for path /")

        try:
            with self.get_driver() as driver:
                driver.get(self.BASE_URL)
                self._handle_popups(driver)
                
                # Search for model
                self._perform_search(driver, model_number)
                
                wait = WebDriverWait(driver, self.timeout)
                wait.until(EC.url_changes(self.BASE_URL))
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                
                self._handle_popups(driver)
                self._clean_page(driver)

                # Handle model page logic
                if "model" in driver.current_url.lower():
                    return self._handle_model_page_search(driver, model_number, part_number, wait)
                else:
                    return ScrapingResult(
                        url=driver.current_url,
                        part_info=PartInfo(
                            partselect_number=part_number,
                            compatibility="Model page not found"
                        ),
                        success=False,
                        error_message="Model page not found"
                    )
                    
        except Exception as e:
            logger.error(f"Error checking compatibility for {model_number}/{part_number}: {e}")
            return ScrapingResult(
                url="",
                part_info=PartInfo(),
                success=False,
                error_message=str(e)
            )

    def _handle_model_page_search(self, driver: webdriver.Chrome, model_number: str, 
                                 part_number: str, wait: WebDriverWait) -> ScrapingResult:
        """Handle search within model page"""
        self._handle_popups(driver)
        
        # Search for part within model page
        part_input = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, self.SELECTORS['part_search_input']))
        )
        part_input.clear()
        part_input.send_keys(part_number)
        
        part_search_btn = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, self.SELECTORS['part_search_button']))
        )
        
        self._handle_popups(driver)
        part_search_btn.click()
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        self._handle_popups(driver)

        # Find and click part link
        try:
            link_element = driver.find_element(By.XPATH, f'//a[contains(@href,"{part_number}")]')
            driver.execute_script("arguments[0].scrollIntoView(true);", link_element)
            self._handle_popups(driver)
            link_element.click()
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            self._handle_popups(driver)
            
            # Extract part information
            part_info = self._extract_part_info(driver)
            part_info.compatibility = "Compatible"
            
            # Save page source
            save_path = self._get_save_path(
                part_number, 
                "model_parts", 
                f"{part_number}_compatibility_{model_number}.txt"
            )
            self._save_content(driver.page_source, save_path, driver.current_url)
            
            return ScrapingResult(
                url=driver.current_url,
                part_info=part_info,
                success=True
            )
            
        except Exception:
            return ScrapingResult(
                url=driver.current_url,
                part_info=PartInfo(
                    partselect_number=part_number,
                    compatibility="Not found"
                ),
                success=False,
                error_message="Part not found on model page"
            )

def main():
    scraper = PartSelectScraper(timeout=15, save_dir="ai_scraper_output")
    
    model_number = "WRS325FDAM04"
    part_number = "WPW10321304"
    
    try:
        result = scraper.get_part_information(model_number)
        if result.success:
            print("Part Information:", result.part_info.to_dict())
        else:
            print("Error:", result.error_message)
        
        compat_result = scraper.check_model_part_compatibility(model_number, part_number)
        if compat_result.success:
            print("Compatibility:", compat_result.part_info.compatibility)
        else:
            print("Compatibility Error:", compat_result.error_message)
            
    except Exception as e:
        logger.error(f"Scraping failed: {e}")

if __name__ == "__main__":
    main()