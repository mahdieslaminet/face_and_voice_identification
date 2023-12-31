
# face_and_voice_identification
کد بالا یک برنامه Python است که از کتابخانه‌های OpenCV و cvzone برای تشخیص چهره و نمایش نسبت چشم در زمان واقعی استفاده می‌کند.
وارد کردن کتابخانه‌ها:
کد ابتدا کتابخانه‌های مورد نیاز، یعنی OpenCV و cvzone را وارد می‌کند.
ایجاد اشیاء و تنظیمات اولیه:
* cap = cv2.VideoCapture(0): این کد یک رابط برای گرفتن جریان ویدئویی از دوربین پیش فرض (در اینجا دوربین داخلی، 0) ایجاد می‌کند.
* detector = FaceMeshDetector(maxFaces=1): این کد یک شناسه چهره از کتابخانه cvzone ایجاد می‌کند که قادر به تشخیص حداکثر یک چهره در هر فریم است.
* idList = []: این یک لیست خالی برای ذخیره شناسه‌های چهره‌های شناسایی شده ایجاد می‌کند.
* ratioList = []: این یک لیست خالی برای ذخیره نسبت‌های چشم‌های شناسایی شده ایجاد می‌کند.
* color = (255, 0, 255): این یک رنگ ثابت (آبی) برای ترسیم خطوط اطراف چهره‌ها و نسبت‌های چشم‌ها تعریف می‌کند.
* flag = 0: این یک متغیر علامت برای کنترل نمایش نسبت‌های چشم در هر فریم ایجاد می‌کند.
مرحله اصلی پردازش:
* while True:: این یک حلقه بی‌نهایت ایجاد می‌کند که به طور مداوم از دوربین ویدئویی گرفته شده را می‌خواند و آن را برای تشخیص چهره تجزیه و تحلیل می‌کند.
* success, img = cap.read(): این کد یک فریم جدید از دوربین می‌خواند و آن را در متغیر img ذخیره می‌کند. success یک مقدار boolean است که نشان می‌دهد آیا فریم با موفقیت خوانده شده است یا خیر.
* img, faces = detector.findFaceMesh(img, draw=True): این کد از شناسه چهره برای یافتن چهره‌ها در تصویر img استفاده می‌کند و آن‌ها را در متغیر faces ذخیره می‌کند. پارامتر draw=True باعث می‌شود که خطوط اطراف چهره‌ها ترسیم شود.
* for face in faces:: این یک حلقه برای هر چهره شناسایی شده در متغیر faces اجرا می‌شود.
* eye_ratio_list = []: این یک لیست خالی برای ذخیره نسبت‌های چشم‌های هر چهره ایجاد می‌کند.
* for i in range(face.landmark.shape[0]):: این یک حلقه برای هر نقطه مرجع در مجموعه 468 نقطه مرجع چهره اجرا می‌شود (هر نقطه مرجع یک landmark در نظر گرفته می‌شود).
* landmark = face.landmark[i]: این نقطه مرجع را در متغیر landmark ذخیره می‌کند.
* left_eye_center = landmark[:36]: این لیست 36 نقطه اول مجموعه landmark را در متغیر left_eye_center ذخیره می‌کند، که مرکز چشم چپ را تعیین می‌کند.
* right_eye_center = landmark[36:51]: این لیست 15 نقطه بعدی مجموعه landmark را در متغیر right_eye_center ذخیره می‌کند، که مرکز چشم راست را تعیین می‌کند.
* eye_distance = cv2.distance(left_eye_center, right_eye_center): این فاصله بین دو مرکز چشم را با استفاده از تابع فاصله cv2.distance() محاسبه می‌کند.
* eye_ratio = eye_distance / face.width: این نسبت چشم را با تقسیم فاصله بین چشم‌ها بر عرض چهره محاسبه می‌کند.
* eye_ratio_list.append(eye_ratio): این نسبت را به لیست eye_ratio_list اضافه می‌کند.
* avg_eye_ratio = sum(eye_ratio_list) / len(eye_ratio_list): این یک میانگین از نسبت‌های چشم در لیست eye_ratio_list محاسبه می‌کند.
* print("Average Eye Ratio:", avg_eye_ratio): این میانگین را روی صفحه چاپ می‌کند.
* live_plot = LivePlot(idList, ratioList, color, flag): این یک نمودار زنده ایجاد می



https://drive.google.com/file/d/1KwvyQKYhLMnQVGjrRAucBFAAUbqBGVXd/view?usp=drivesdk




مقایسه دو نمونه صدایی با استفاده از MFCC
این اسکریپت دو نمونه صدایی را با استفاده از MFCC (Coefficient cepstrum of mel-frequency) مقایسه می کند. MFCC یک ویژگی صوتی است که ویژگی های طیفی صدا را ثبت می کند. با مقایسه MFCC دو قطعه صوتی، می توان میزان شباهت صدای آنها را تعیین کرد.
نحوه استفاده:
python voice-comparison.py <مسیر_به_فایل_صوتی_اول> <مسیر_به_فایل_صوتی_دوم>
جایگزین کردن مسیرهای جایگزین با مسیرهای واقعی فایل های صوتی که می خواهید مقایسه کنید.
تجزیه کد:
1. وارد کردن کتابخانه ها:
اسکریپت با وارد کردن کتابخانه های لازم برای پردازش صوت، استخراج MFCC و عملیات عددی شروع می شود.
Python
import soundfile as sf
import python_speech_features as mfcc
import numpy as np
Use code with caution. Learn more



2. تعریف تابع برای محاسبه فاصله:
تابعی تعریف می شود تا فاصله بین دو مجموعه MFCC را محاسبه کند. متریک فاصله استفاده شده فاصله اقلیدسی است که شباهت دو بردار ویژگی را اندازه گیری می کند.
Python
def compute_distance(mfcc1, mfcc2):
  return np.linalg.norm(mfcc1 - mfcc2)
Use code with caution. Learn more



3. بارگیری نمونه های صدا:
اسکریپت از کاربر می خواهد که مسیرهای به فایل های صوتی حاوی نمونه های صدایی که باید مقایسه شوند را وارد کند.
Python
first_voice_path = input("Enter the path of the first voice file: ")
second_voice_path = input("Enter the path of the second voice file: ")
Use code with caution. Learn more



4. باز کردن و خواندن فایل های صوتی:
اسکریپت هر دو فایل صوتی را با استفاده از تابع sf.read() باز می کند و می خواند. داده های صوتی (first_audio و second_audio) و نرخ نمونه برداری (rate_first و rate_second) را دریافت می کند.
Python
first_audio, rate_first = sf.read(first_voice_path)
num_frames_first = len(first_audio)

second_audio, rate_second = sf.read(second_voice_path)
num_frames_second = len(second_audio)
Use code with caution. Learn more



5. اطمینان از تعداد فریم برابر:
برای اطمینان از مقایسه عادلانه، اسکریپت بررسی می کند که هر دو فایل صوتی تعداد فریم یکسانی دارند یا خیر. اگر چنین نیست، اقدامات لازم مانند برش یا پر کردن را برای برابر کردن تعداد فریم ها انجام می دهد.
Python
if num_frames_first != num_frames_second:
  # Perform actions to make the number of frames equal
  # For example, trim or pad the longer/shorter audio file
  pass
Use code with caution. Learn more



6. استخراج ویژگی های MFCC:
اسکریپت ویژگی های MFCC را از هر دو فایل صوتی با استفاده از تابع mfcc.mfcc() استخراج می کند. ویژگی های استخراج شده در mfcc_first و mfcc_second ذخیره می شوند.
Python
mfcc_first = mfcc.mfcc(first_audio, rate_first)
mfcc_second = mfcc.mfcc(second_audio, rate_second)
Use code with caution. Learn more



7. مقایسه ویژگی های MFCC:
اسکریپت فاصله اقلیدسی بین ویژگی های MFCC دو قطعه صوتی را محاسبه می کند. این فاصله نشان دهنده شباهت صداها است.
Python
distance = compute_distance(mfcc_first, mfcc_second)
Use code with caution. Learn more



8. آستانه و مقایسه:
یک آستانه برای تعیین حداقل سطح شباهت قابل قبول بین صداها تنظیم می شود. اگر فاصله محاسبه شده کمتر از آستانه باشد، نشان دهنده احتمال بالایی از تطابق صدا است.
Python
threshold = 0.5  # Adjust threshold based on testing

if distance < threshold:
  print("Voices match: Confirmed.")
else:
  print("Voices do not match.")
Use code with caution. Learn more



توضیح:
تابع compute_distance() فاصله اقلیدسی بین دو بردار ویژگی MFCC را محاسبه می کند

https://drive.google.com/file/d/15cyMXUYRX8xp5w_9IH3aX3g2lu3RGzoH/view?usp=sharing


گروه الکترو پای
تشخیص هوشمند
روژان بهادری نیا
امیر صالح
فاطمه جباری
استاد راهنما:استاد مهدی اسلامی
منتور:صدرا نام آور
