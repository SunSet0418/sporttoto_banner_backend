from flask import Blueprint, request as req, jsonify
from selenium import webdriver
from Server.utils import utils
import os

app = Blueprint('image', __name__)

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('headless')
chrome_options.add_argument('no-sandbox')

driver = webdriver.Chrome(executable_path=os.getcwd()+"/Server/utils/"+utils.check_os_type()+"_chromedriver", chrome_options=chrome_options)

driver.implicitly_wait(3)

@app.route('/', methods=['GET'])
def image_main():
    return "<h1>Image Route</h1>"


@app.route('/check', methods=['POST'])
def image_check():
    images = []

    r_url = req.form['url']

    driver.get(r_url)

    image_elements = driver.find_elements_by_tag_name('img')

    for element in image_elements:

        images.append(element.get_attribute('src'))

    print("Detected Images : "+str(len(images)))

    print(images)

    for image in images:
        utils.check_toto_banner(image)

    # driver.quit()

    return jsonify(images)








