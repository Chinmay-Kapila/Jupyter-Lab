from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time

driver=webdriver.Chrome()
driver.get("https://www.python.org/")

element= driver.find_element(By.NAME, "p") 
element.send_keys("pip" + Keys.RETURN)

time.sleep(5)

