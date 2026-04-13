from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

def train_model(df):
    # ফিগার (X) এবং টার্গেট (y) আলাদা করা
    # ধরে নিচ্ছি আপনার ডেটাসেটের শেষ কলামটি হলো টার্গেট বা লেবেল
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # ট্রেনিং এবং টেস্টিং সেটে ভাগ করা
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("মডেল ট্রেনিং শুরু হচ্ছে...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # প্রেডিকশন করা
    y_pred = model.fit(X_train, y_train).predict(X_test)

    print("\n--- মডেল ইভালুয়েশন রিপোর্ট ---")
    # এখানে zero_division=0 যোগ করা হয়েছে লাল ওয়ার্নিং বন্ধ করার জন্য
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print(f"সামগ্রিক নির্ভুলতা (Accuracy): {accuracy_score(y_test, y_pred):.2f}")
    
    return model, X_test, y_test, y_pred