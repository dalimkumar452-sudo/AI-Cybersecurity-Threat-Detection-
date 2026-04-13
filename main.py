from src.data_preprocessing import load_data, preprocess_data
from src.model import train_model
from src.visualize import create_visualizations # আগের কোডটি visualize.py তে থাকলে

def main():
    print("১. ডেটাসেট লোড করা হচ্ছে...")
    df = load_data("data/kdd.csv")
    
    print("২. ডেটা প্রি-প্রসেসিং হচ্ছে...")
    df = preprocess_data(df)
    
    # ৩. মডেল ট্রেইন এবং ডেটা সংগ্রহ
    model, X_test, y_test, y_pred = train_model(df)
    
    # ৪. LinkedIn এর জন্য গ্রাফ তৈরি করা
    print("৩. LinkedIn এর জন্য গ্রাফ তৈরি করা হচ্ছে...")
    create_visualizations(y_test, y_pred, model, df.iloc[:, :-1].columns)
    
    print("\nসব কাজ শেষ! আপনার ফোল্ডারে confusion_matrix.png এবং feature_importance.png দেখুন।")

if __name__ == "__main__":
    main()