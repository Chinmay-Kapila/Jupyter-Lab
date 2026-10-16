from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

brave_path = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"

options = Options()
options.binary_location = brave_path

driver = webdriver.Chrome(
    service=Service(
        ChromeDriverManager(driver_version="144.0.7559.133").install()
    ),
    options=options
)

driver.get("https://google.com")
