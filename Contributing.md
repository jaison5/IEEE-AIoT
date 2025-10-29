# 🤝 Contributing Guidelines

Thank you for your interest in contributing to *Robotic-AIoT-Max*!  
This project aims to build a modular AIoT framework for low-latency real-time robotic teleoperation, integrating vision (YOLOv11-nano), embedded controllers (ESP8266), and edge devices.  
We welcome all kinds of contributions — bug reports, new features, dataset improvements, documentation enhancements, or deployment optimizations.

---

## 🧰 How to Contribute

1. **Fork this repository**  
   Click the “Fork” button at the top right of the repository page.

2. **Clone your fork locally**  
   ```bash
   git clone https://github.com/your-username/Robotic-AIoT-Max.git
   cd Robotic-AIoT-Max

3. **Create a new branch for your change**
   ```bash
   git checkout -b feature/my-feature

4. **Make your changes**
- For code changes: update files under `/code`, `/datasets`, `/final model`, or other relevant folders.
- For dataset or model updates: update YAML or dataset files under `/datasets/20250105`, update `train2.py` or other scripts accordingly.
- For documentation: update README.md, include relevant links, images, or examples.
5. **Commit your changes**
  ```bash
  git add .
  git commit -m "Add feature: description"
  ```
6. **Push your branch and open a Pull Request (PR)**
   ```bash
   git push origin feature/my-feature
   ```

Then open a PR to the `main` branch of this repository. In your PR description, please explain:
- What you changed
- Why the change is needed
- Any additional steps to test your change

---

## 🧩 Types of Contributions
- 🐞 Bug fixes
- ✨ New features (e.g., new vision classes, robotic arm behaviors, edge deployment formats)
- 📊 Performance optimizations (e.g., faster inference, smaller model footprint)
- 🧾 Documentation updates (e.g., clearer instructions, new example images)
- 📁 Dataset improvements (e.g., adding new classes, cleaning annotations)
- 🔧 Integration work (e.g., adding support for new hardware or protocols)

---

## 🔧 Development Setup
- Environment: Python 3.10+, PyTorch, Ultralytics YOLOv11-nano
- Embedded controller: ESP8266 using Arduino IDE / ESP-IDF, I2C & Serial communication
- Frontend controller: Flutter (Dart), LiveKit video streaming, MQTT topics
- Dataset directory: `/datasets/20250105`
Training and evaluation scripts: `train2.py`, `01.train.py`, `/code folder`

---

## ✅ Pull Request Checklist
Before submitting your PR, please ensure:
- [ ] Your changes are described comprehensively in the PR description
- [ ]Code follows the existing style and conventions
- [ ]Any new dependencies are documented in README.md or requirements.txt
- [ ]All existing tests (if any) pass
- [ ]For dataset or model changes: updated performance metrics or comparative results included
- [ ]Documentation updated (e.g., images, tables, README sections)
- [ ]You have signed your commits with a valid DCO (if required)

---

## ⚖️ License & Contribution Agreement
By submitting a contribution to this project, you agree that your contributions will be licensed under the [MIT License](https://github.com/jaison5/Robotic-AIoT-Max/blob/main/LICENSE) of this repository.

---

## 🪪 Code of Conduct
All contributors are expected to follow respectful, inclusive, and professional behaviour.
Please treat others with kindness and respect — harassment or discrimination of any kind will not be tolerated.

---

Thank you for contributing to Robotic-AIoT-Max! Your support and collaboration make this project better for everyone.
