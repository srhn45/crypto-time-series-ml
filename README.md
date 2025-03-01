# Crypto/Time Series ML Hyperparameter Optimizer Tool

A tool for performing systematic hyperparameter optimization for time-series data. With the example data and model architecture, I use a modular LSTM-CNN-MLP based model to forecast BTC price movements 1 hour in advance.

## Overview

This project provides a flexible framework for iterative hyperparameter search and model training. Finding the correct model architecture for a specific problem can be time consuming, I aim to speed up the process for researchers and developers with this tool. It's designed to work with custom model architectures and can optimize any arbitrary hyperparameter, making it adaptable for various deep learning frameworks and specific use cases.

The tool also includes a custom training function with a sliding window for financial data that attempts to recreate real market circumstances.
The latest data is always used for testing to train for the latest market dynamics.

The search space can be modified directly inside the search function.

I didn't implement a separate test set since there already wasn't a lot of data. However, implementing it should be trivial for anyone interested.


## Prerequisites
- Python 3.8+
- TensorFlow 2.x
- pandas
- numpy
- python-binance (for crypto data)

## Installation

```bash
git clone https://github.com/srhn45/crypto-time-series-ml.git
cd crypto-time-series-ml
pip install -r requirements.txt

## **Usage**  
1. **Install dependencies**:  

2. **Run the search tool**:
   python main.py

3. Modify train.py, model.py, etc. to implement custom architectures, use different datasets or modify the search space.
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{crypto-time-series-ml,
  author = {srhn45},
  year = {2025},
  url = {https://github.com/srhn45/crypto-time-series-ml}
}
```

## Contact

- GitHub: [@srhn45](https://github.com/srhn45)
- Project Link: [https://github.com/srhn45/crypto-time-series-ml](https://github.com/srhn45/crypto-time-series-ml)
