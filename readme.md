# 🎵 Spotify Music Recommendation System

## 📁 Cấu trúc thư mục
```
spotify-recommender/
├── 📁 data/
│   ├── spotify_million_playlist_dataset/     # Dataset gốc (1M playlists)
│   └── spotify_million_playlist_dataset_challenge/
├── 📁 DeTai1_Spotify/
│   └── Spotify_test.json                     # Test data cho submission
├── 📁 docker/
│   ├── docker-compose.yml                    # Docker configuration
│   └── hadoop-hive.env
├── 📁 src/
│   ├── preprocess_data.py                    # Tiền xử lý dữ liệu
│   ├── train_model.py                        # Huấn luyện ALS model
│   ├── evaluate_model.py                     # Đánh giá MAP@100
│   ├── generate_submission.py                # Tạo file submission.json
│   └── upload_to_hdfs.py
├── 📁 output/
│   └── submission.json                       # Kết quả cuối cùng
└── *.bat                                     # Scripts chạy từng bước
```

## ⚙️ Cấu hình hệ thống
- **Docker**: Hadoop 3.3.1 + Spark 3.1.1
- **Memory**: 14GB Spark Worker
- **Algorithm**: Hybrid ALS + Popularity (70% + 30%)

## 🚀 Hướng dẫn chạy

### 1. Khởi động hệ thống
```batch
start_system.bat                              
```

### 2. Upload và xử lý dữ liệu  
```batch
upload_data.bat                               
run_preprocess.bat                            
```

### 3. Huấn luyện và đánh giá
```batch
run_train.bat                                 
evaluate_model.bat                            
```

### 4. Tạo submission
```batch
run_submission.bat                          
```

## 📊 Kết quả đạt được
- **MAP@100**: 0.041233 
- **Model**: Hybrid ALS + Popularity
- **Format**: 100 tracks/playlist theo yêu cầu đề bài

## 📝 Files quan trọng
- **Input**: `DeTai1_Spotify/Spotify_test.json`
- **Output**: `output/submission.json` 
- **Model**: HDFS `/spotify_data/processed/model/`

**🎯 Hệ thống ready cho submission với MAP@100 = 0.041233!**
