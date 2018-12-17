# CocktailClassification
ระบบจำแนกคอกเทลเพื่อบอกส่วนผสมและปริมาณแอลกอฮอล์เป็นทางเลือกสำหรับผู้ที่สนใจด้วย Deep Learning

# แนวคิด
คอกเทลมีมากมายหลายชนิด ถ้าหากเป็นผู้ที่สนใจก็จะจำแนกได้ไม่ยากด้วยตาเปล่า แต่หากใช้ Computer vision และ Deep Learning ในการตรวจสอบและจำแนก \
มันจะเวิร์คมั้ยนะ นี่คือสิ่งที่ผมคิดตามแบบฉบับคนชอบแอลกอฮอล์เป็นชีวิตจิตใจ

เริ่มต้น ผมจะใช้ Cocktail 10 ชนิดดังนี้

-B52

![alt text](https://media-cdn.tripadvisor.com/media/photo-s/0a/eb/68/0c/b52-cocktail-on-fire.jpg)
-Black Russian


![alt text](https://i0.wp.com/www.himbuds.com/wp-content/uploads/2017/01/Black-Russian.jpg?fit=600%2C315)


-Bloody mary


![alt text](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQeKffZ4GWJtQaE_MW4AxNVY4G7Wm7KuZDVPX340wbO8jaVeZPZRA)


-Blue Hawaii


![alt text](https://bitzngiggles.com/wp-content/uploads/2018/05/Blue-Hawaiian-photo.jpg)


-Dry Martini


![alt text](https://i.pinimg.com/originals/84/5a/79/845a790de7cd56b49019ad1d9d94b062.jpg)


-Midori Sour


![alt text](https://cdn.liquor.com/wp-content/uploads/2017/02/01121823/midori-sour-720x720-recipe.jpg)


-Mojito


![alt text](https://www.jamieoliver.com/drinks-tube/wp-content/uploads/2014/06/Mojito.jpg)


-Pina Colada


![alt text](http://www.gourmetgadgetry.com/wp-content/uploads/2015/01/Pina-Colada.jpg)


-Screw Driver


![alt text](https://www.ndtv.com/cooks/images/screwdriver-620.png)


-Sex on the beach


![alt text](https://andpour.com/media/catalog/product/cache/3e0e5d06d2a14e7c80c1b508d42bc248/a/p/ap_sex-on-the-beach.jpg)

แล้วใช้ CNN ในการเทรน โดยการผ่าน Convolutional Layer สลับกับ Max Pooling 4 รอบ จากนั้นผ่าน Hidden Layer 64 รอบนึง และ 32 รอบนึง
ผลปรากฎว่าการทำนายไม่ดีเท่าที่ควร อยู่ที่ 81% 

จึงได้ใส่ Initialization และ Regularization เพื่อช่วยกระตุ้นการลด Loss (ได้ลองใส่ Dropout ลงไปแล้ว แต่ผลแย่ลง จึงตัดออกไป)
ผลปัจจุบันตอนนี้อยู่ที่ 87% เมื่อลด Learning Rate ให้เหลือ 0.001 และเพิ่ม Epoch เป็น 30 ก็ช่วยให้มีความแม่นยำมากขึ้นจริง
https://drive.google.com/file/d/14vntzjitQrccAVL-oO1LUxEO6zRfoOxE/view?usp=sharing


วิธีการใช้งานโปรแกรม
อย่างแรกให้ทำการโหลด Data set จากลิ๊ง (ผม Label เอง)
https://drive.google.com/file/d/1MWy1hQ0UosfAMxJOvMwm6Iydqi0ijjd6/view?usp=sharing
จากนั้นแตกไฟล์ที่ที่อยู่เดียวกับโค้ด CocktailClassification.py 
จากนั้นทำการรันดูใน spyder 

หากสามารถใช้ได้ ก็จะสามารถทำนายด้วยตัวเองเล่นๆได้โดยการเปลี่ยนที่อยู่ของรูปที่ต้องการทำนาย ในโค้ด Input ด้านล่างโมเดล


# สรูป
เอาจริงๆ ผมเองก็ไม่รู้ว่าจะเอาไปทำอะไรได้ต่อไป แต่อย่างน้อยหากเรารู้ส่วนผสมและปริมาณแอลกอฮอล์ในเครื่องดื่มที่เรากิน
เราจะได้เอาไปคำนวณได้ว่า จะกินอีกกี่แก้วดีนะ ที่จะทำให้เป่าเครื่องเป่าที่ด่านแล้วจะไม่ติดคุก 5555555

