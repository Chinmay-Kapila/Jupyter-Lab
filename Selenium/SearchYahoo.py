from selenium import webdriver
from selenium.webdriver. common.keys import Keys
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome()
driver.get("https://opensubtitles.org/en/search/sublanguageid-all/moviename-u")

# element = driver. find_element(By.NAME, "MovieName")
# element.send_keys("Cars"+Keys.RETURN)

# # element = driver.find_elements(By.TAG_NAME, 'li')
# element = driver. find_element(By. CLASS_NAME,'searchCenterMiddle' )
# print(element.text)
# print(element.get_attribute('outerHTML'))
# print(type(element))
# time.sleep(5)

# element = driver.find_element(By.ID,'logo')
# element.click()

time.sleep(3)

# driver.refresh()
# driver.back()
# time.sleep(2)
# driver.forward()
# time.sleep(4)

driver.quit()

