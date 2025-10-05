import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn

# ایجاد پارامترهای شناخته‌شده
weight = 0.7
bias = 0.3

# ایجاد داده‌ها (ورودی‌ها و خروجی‌ها)
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
Y = weight * X + bias
# print(X[:10], Y[:10], len(X), len(Y))

# تقسیم داده‌ها به مجموعه‌ی آموزش و تست (یکی از مفاهیم بسیار مهم در یادگیری ماشین)
train_split = int(0.8 * len(X))
X_train, Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split:], Y[train_split:]
# print(len(X_train), len(Y_train), len(X_test), len(Y_test))

# تابعی برای رسم داده‌ها و پیش‌بینی‌ها
def plot_predictions(train_data=X_train,
                     train_labels=Y_train,
                     test_data=X_test,
                     test_labels=Y_test,
                     predictions=None):
    plt.figure(figsize=(10,10))

    # نمایش داده‌های آموزشی با رنگ آبی
    plt.scatter(train_data, train_labels, c="b", s=4, label="داده‌های آموزشی")

    # نمایش داده‌های تست با رنگ سبز
    plt.scatter(test_data, test_labels, c="g", s=4, label="داده‌های تست")

    # اگر پیش‌بینی وجود داشته باشد، آن را با رنگ قرمز رسم کن
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="پیش‌بینی‌ها")

    plt.legend(prop={'size':14})

plot_predictions()
plt.show()

# ایجاد یک کلاس مدل رگرسیون خطی
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # تعریف وزن و بایاس به‌عنوان پارامترهای قابل یادگیری
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    # تابع forward (نحوه‌ی محاسبه‌ی خروجی مدل)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


# ثابت‌سازی مقدار تصادفی‌ها برای تکرارپذیری نتایج
torch.manual_seed(42)

# ایجاد یک نمونه از مدل (زیرکلاس nn.Module)
model_0 = LinearRegression()

# نمایش وضعیت اولیه‌ی پارامترها
model_0.state_dict()

# انجام پیش‌بینی اولیه با مدل (قبل از آموزش)
with torch.inference_mode():
    y_preds = model_0(X_test)
print(y_preds)

# رسم داده‌ها و پیش‌بینی اولیه
plot_predictions(predictions=y_preds)
plt.show()

# تعریف تابع خطا (Loss Function)
loss_fn = nn.L1Loss()  # خطای L1، یعنی میانگین قدر مطلق اختلاف بین پیش‌بینی و مقدار واقعی

# تعریف بهینه‌ساز (Optimizer)
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)  # الگوریتم گرادیان نزولی تصادفی

# تعداد epochها (هر epoch یعنی یک بار عبور کامل از کل داده‌ها)
epochs = 200

# شروع فرآیند آموزش
for epoch in range(epochs):
    # تنظیم مدل در حالت آموزش
    model_0.train()  # در این حالت، گرادیان‌ها محاسبه و به‌روزرسانی می‌شوند

    # گام ۱: انجام پیش‌بینی روی داده‌های آموزش
    ypred = model_0(X_train)

    # گام ۲: محاسبه‌ی مقدار خطا
    loss = loss_fn(ypred, Y_train)

    # گام ۳: صفر کردن گرادیان‌های قبلی (برای جلوگیری از جمع شدن گرادیان‌ها)
    optimizer.zero_grad()

    # گام ۴: محاسبه‌ی گرادیان‌ها با انجام backpropagation
    loss.backward()

    # گام ۵: به‌روزرسانی وزن‌ها و بایاس با استفاده از گرادیان‌ها
    optimizer.step()

    # چاپ وضعیت هر epoch (برای مشاهده روند کاهش خطا)
    if epoch % 1 == 0:
        print(
            f"Epoch {epoch + 1}: Loss = {loss.item():.4f}, Weight = {model_0.weights.item():.4f}, Bias = {model_0.bias.item():.4f}"
        )

    # تنظیم مدل در حالت ارزیابی (غیرفعال کردن گرادیان‌ها)
    model_0.eval()

# ---- بعد از اتمام آموزش ----
model_0.eval()  # مدل را در حالت ارزیابی قرار می‌دهیم

# انجام پیش‌بینی جدید با مدل آموزش‌دیده
with torch.inference_mode():
    y_preds_new = model_0(X_test)

# رسم نمودار داده‌های واقعی و پیش‌بینی‌ها (نقاط قرمز، خروجی مدل آموزش‌دیده)
plot_predictions(predictions=y_preds_new)
plt.show()
