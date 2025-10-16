#include <Wire.h>
#include <Servo.h>
#include <AccelStepper.h>
#include <MultiStepper.h>
#include <math.h>

// =================================================================================
// <<< 新增的全域變數 >>>
// volatile 關鍵字是必須的，因為此變數會在中斷(receiveEvent)中被修改
volatile bool newDataFromI2C = false; 
// 用於儲存從 I2C 完整接收到的指令
String i2cCommand = "";
// =================================================================================


// Define stepper pins
#define ASTEP_PIN 48  // Step pin
#define ADIR_PIN 50   // Direction pin
#define AENABLE 52
#define BSTEP_PIN 44  // Step pin
#define BDIR_PIN 46   // Direction pin
#define CSTEP_PIN 40  // Step pin
#define CDIR_PIN 42   // Direction pin
#define DSTEP_PIN 2   // Step pin
#define DDIR_PIN 5    // Direction pin
#define ESTEP_PIN 3   // Step pin
#define EDIR_PIN 6
#define FSTEP_PIN 4
#define FDIR_PIN 7
#define ENABLE 8

#define Aendstop 43
#define Bendstop 41
#define Cendstop 39
#define Dendstop 37
#define Eendstop 35
#define emergbtn 53

const int SLAVE_ADDRESS = 1;
char incomingByte = 0;

String i2cC = "";

float alocation = 0;
float blocation = 0;
float clocation = 0;
float dlocation = 0;
float elocation = 0;
float flocation = 0;
float clawlocation = 0;

long positions[6];

String I2Creturn = "";

// bool I2Cin = 0; // <<< 這個舊旗標不再需要，由 newDataFromI2C 取代

const float agear = 1;
const float bgear = 50;
const float cgear = 27;
const float dgear = 20;
const float egear = 1;
const float fgear = 1;
const float steppower = 4.4;  // 1/(360/steps/rev)
const float radtoang = 57.295;


// Steps per revolution for the motor
const float speed = 50;
const float maxspeed = 75;

const float accel = 500;
const float stepsPerRevolution = 1600;
// Microstepping multiplier (1, 2, 4, 8, 16, or 32)

// AccelStepper instance in driver mode
AccelStepper stepperA(AccelStepper::DRIVER, ASTEP_PIN, ADIR_PIN);
AccelStepper stepperB(AccelStepper::DRIVER, BSTEP_PIN, BDIR_PIN);
AccelStepper stepperC(AccelStepper::DRIVER, CSTEP_PIN, CDIR_PIN);
AccelStepper stepperD(AccelStepper::DRIVER, DSTEP_PIN, DDIR_PIN);
AccelStepper stepperE(AccelStepper::DRIVER, ESTEP_PIN, EDIR_PIN);
AccelStepper stepperF(AccelStepper::DRIVER, FSTEP_PIN, FDIR_PIN);
MultiStepper robot2T;
Servo claw;

static const int DOF = 6;
static const float alpha_[DOF] = { -1.57f, 0.00f, 1.57f, -1.57f, 1.57f, 0.00f };     // radians
static const float a_[DOF] = { 0.000f, 252.625f, 0.000f, 0.000f, 0.000f, 0.000f };   // mm
static const float d_[DOF] = { 136.1f, 0.0f, 0.0f, 214.6f, 0.0f, 129.471f };         // mm
static const float theta_off_[DOF] = { 0.00f, -1.57f, 1.57f, 0.00f, 0.00f, 0.00f };  // radians

// IK settings
static const int MAX_ITERS = 120;     // iterations cap
static const float DAMPING = 0.02f;   // lambda for DLS (radians & mm units)void
static const float POS_TOL = 0.5f;    // mm
static const float ROT_TOL = 0.005f;  // rad (~0.29 deg)
// ---------------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------------
struct Vec3 {
  float x, y, z;
};

static inline float clampf(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

static inline void mat4Mul(const float A[16], const float B[16], float C[16]) {
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      float s = 0;
      for (int k = 0; k < 4; k++) s += A[r * 4 + k] * B[k * 4 + c];
      C[r * 4 + c] = s;
    }
  }
}

static inline void mat4Copy(const float S[16], float D[16]) {
  for (int i = 0; i < 16; i++) D[i] = S[i];
}

static inline void mat4FromDH(float a, float alpha, float d, float theta, float T[16]) {
  float ct = cosf(theta), st = sinf(theta);
  float ca = cosf(alpha), sa = sinf(alpha);
  T[0] = ct;
  T[1] = -st * ca;
  T[2] = st * sa;
  T[3] = a * ct;
  T[4] = st;
  T[5] = ct * ca;
  T[6] = -ct * sa;
  T[7] = a * st;
  T[8] = 0;
  T[9] = sa;
  T[10] = ca;
  T[11] = d;
  T[12] = 0;
  T[13] = 0;
  T[14] = 0;
  T[15] = 1;
}

static inline void mat4ExtractRPY_ZYX(const float T[16], float &rx, float &ry, float &rz) {
  // ZYX intrinsic (yaw Z, pitch Y, roll X). Using rotation matrix R = T[0..10].
  const float r00 = T[0], r01 = T[1], r02 = T[2];
  const float r10 = T[4], r11 = T[5], r12 = T[6];
  const float r20 = T[8], r21 = T[9], r22 = T[10];
  float pitch = asinf(-clampf(r20, -1.0f, 1.0f));
  float roll, yaw;
  if (fabsf(r20) < 0.9999f) {
    roll = atan2f(r21, r22);
    yaw = atan2f(r10, r00);
  } else {
    // Gimbal-close case: use alternative
    roll = 0.0f;
    yaw = atan2f(-r01, r11);
  }
  rx = roll;
  ry = pitch;
  rz = yaw;
}

static inline void rpyZYX_to_R(float rx, float ry, float rz, float R[9]) {
  // rx=roll(X), ry=pitch(Y), rz=yaw(Z)
  float cx = cosf(rx), sx = sinf(rx);
  float cy = cosf(ry), sy = sinf(ry);
  float cz = cosf(rz), sz = sinf(rz);
  // R = Rz(rz)*Ry(ry)*Rx(rx)
  R[0] = cz * cy;
  R[1] = cz * sy * sx - sz * cx;
  R[2] = cz * sy * cx + sz * sx;
  R[3] = sz * cy;
  R[4] = sz * sy * sx + cz * cx;
  R[5] = sz * sy * cx - cz * sx;
  R[6] = -sy;
  R[7] = cy * sx;
  R[8] = cy * cx;
}

static inline void R_to_axisAngle(const float R[9], float axis[3], float &angle) {
  float tr = R[0] + R[4] + R[8];
  float c = (tr - 1.0f) * 0.5f;
  c = clampf(c, -1.0f, 1.0f);
  angle = acosf(c);
  if (angle < 1e-6f) {
    axis[0] = axis[1] = axis[2] = 0;
    angle = 0;
    return;
  }
  float denom = 2.0f * sinf(angle);
  axis[0] = (R[7] - R[5]) / denom;
  axis[1] = (R[2] - R[6]) / denom;
  axis[2] = (R[3] - R[1]) / denom;
}

// ---------------------------------------------------------------------------------
// Forward Kinematics
// ---------------------------------------------------------------------------------
static void fk(const float q[DOF], float T[16]) {
  float Ti[16], Tacc[16];
  // Init to identity
  for (int i = 0; i < 16; i++) Tacc[i] = (i % 5 == 0) ? 1.0f : 0.0f;
  for (int i = 0; i < DOF; i++) {
    float theta = q[i] + theta_off_[i];
    mat4FromDH(a_[i], alpha_[i], d_[i], theta, Ti);
    float Tnew[16];
    mat4Mul(Tacc, Ti, Tnew);
    mat4Copy(Tnew, Tacc);
  }
  mat4Copy(Tacc, T);
}

// ---------------------------------------------------------------------------------
// Jacobian (geometric) for 6R arm
// ---------------------------------------------------------------------------------
static void jacobian(const float q[DOF], float J[6 * DOF]) {
  float Tacc[16];
  // identity
  for (int i = 0; i < 16; i++) Tacc[i] = (i % 5 == 0) ? 1.0f : 0.0f;

  float origins[DOF + 1][3];
  float zaxes[DOF + 1][3];

  // base frame
  origins[0][0] = 0;
  origins[0][1] = 0;
  origins[0][2] = 0;
  zaxes[0][0] = 0;
  zaxes[0][1] = 0;
  zaxes[0][2] = 1;

  // accumulate frames
  for (int i = 0; i < DOF; i++) {
    float Ti[16];
    float theta = q[i] + theta_off_[i];
    mat4FromDH(a_[i], alpha_[i], d_[i], theta, Ti);
    float Tnew[16];
    mat4Mul(Tacc, Ti, Tnew);
    mat4Copy(Tnew, Tacc);

    origins[i + 1][0] = Tacc[3];
    origins[i + 1][1] = Tacc[7];
    origins[i + 1][2] = Tacc[11];

    zaxes[i + 1][0] = Tacc[2];
    zaxes[i + 1][1] = Tacc[6];
    zaxes[i + 1][2] = Tacc[10];
  }

  float pe[3] = { origins[DOF][0], origins[DOF][1], origins[DOF][2] };

  for (int i = 0; i < DOF; i++) {
    float zi[3] = { zaxes[i][0], zaxes[i][1], zaxes[i][2] };
    float pi[3] = { origins[i][0], origins[i][1], origins[i][2] };
    float r[3] = { pe[0] - pi[0], pe[1] - pi[1], pe[2] - pi[2] };
    // cross(zi, r)
    float jv[3] = { zi[1] * r[2] - zi[2] * r[1], zi[2] * r[0] - zi[0] * r[2], zi[0] * r[1] - zi[1] * r[0] };

    J[0 * DOF + i] = jv[0];
    J[1 * DOF + i] = jv[1];
    J[2 * DOF + i] = jv[2];
    J[3 * DOF + i] = zi[0];
    J[4 * DOF + i] = zi[1];
    J[5 * DOF + i] = zi[2];
  }
}

// Multiply 6x6 with 6x1: y = A*x
static inline void mat6x6_mul_vec(const float A[36], const float x[6], float y[6]) {
  for (int r = 0; r < 6; r++) {
    float s = 0;
    for (int c = 0; c < 6; c++) s += A[r * 6 + c] * x[c];
    y[r] = s;
  }
}

// Build normal matrix A = J*J^T + (lambda^2) I (6x6) and compute its inverse via adjugate/Gauss-Jordan (simple, not super-optimized)
// Returns false if singular.
static bool invert6x6(const float A[36], float invA[36]) {
  // Augment [A | I] and perform Gauss-Jordan
  float M[6][12];
  for (int r = 0; r < 6; r++) {
    for (int c = 0; c < 6; c++) M[r][c] = A[r * 6 + c];
    for (int c = 0; c < 6; c++) M[r][6 + c] = (r == c) ? 1.0f : 0.0f;
  }
  for (int i = 0; i < 6; i++) {
    // pivot
    int piv = i;
    float maxA = fabsf(M[i][i]);
    for (int r = i + 1; r < 6; r++) {
      float v = fabsf(M[r][i]);
      if (v > maxA) {
        maxA = v;
        piv = r;
      }
    }
    if (maxA < 1e-8f) return false;
    if (piv != i) {
      for (int c = 0; c < 12; c++) {
        float tmp = M[i][c];
        M[i][c] = M[piv][c];
        M[piv][c] = tmp;
      }
    }
    // normalize row
    float diag = M[i][i];
    for (int c = 0; c < 12; c++) M[i][c] /= diag;
    // eliminate
    for (int r = 0; r < 6; r++)
      if (r != i) {
        float f = M[r][i];
        if (f != 0) {
          for (int c = 0; c < 12; c++) M[r][c] -= f * M[i][c];
        }
      }
  }
  // extract inv
  for (int r = 0; r < 6; r++)
    for (int c = 0; c < 6; c++) invA[r * 6 + c] = M[r][6 + c];
  return true;
}

// Compute error twist between current T(q) and desired pose (x,y,z, rpy_ZYX)
static void poseError(const float Tcurr[16], const float p_des[3], const float rpy_des[3], float e[6]) {
  // position error
  e[0] = p_des[0] - Tcurr[3];
  e[1] = p_des[1] - Tcurr[7];
  e[2] = p_des[2] - Tcurr[11];
  // orientation error via axis-angle of R_err = R_des * R_curr^T
  float Rcurr[9] = { Tcurr[0], Tcurr[1], Tcurr[2], Tcurr[4], Tcurr[5], Tcurr[6], Tcurr[8], Tcurr[9], Tcurr[10] };
  float Rdes[9];
  rpyZYX_to_R(rpy_des[0], rpy_des[1], rpy_des[2], Rdes);
  // R_err = Rdes * Rcurr^T
  float Rct[9] = { Rcurr[0], Rcurr[3], Rcurr[6], Rcurr[1], Rcurr[4], Rcurr[7], Rcurr[2], Rcurr[5], Rcurr[8] };
  float Rerr[9];
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++) {
      float s = 0;
      for (int k = 0; k < 3; k++) s += Rdes[r * 3 + k] * Rct[k * 3 + c];
      Rerr[r * 3 + c] = s;
    }
  float axis[3];
  float ang;
  R_to_axisAngle(Rerr, axis, ang);
  e[3] = axis[0] * ang;
  e[4] = axis[1] * ang;
  e[5] = axis[2] * ang;  // small-angle rotation vector
}

// Damped Least Squares IK: returns true if converged, q is updated in-place
static bool ik_solve(float q[DOF], const float p_des[3], const float rpy_des[3]) {
  for (int iter = 0; iter < MAX_ITERS; ++iter) {
    // FK and error
    float Tcurr[16];
    fk(q, Tcurr);
    float e[6];
    poseError(Tcurr, p_des, rpy_des, e);

    float pos_err = sqrtf(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    float rot_err = sqrtf(e[3] * e[3] + e[4] * e[4] + e[5] * e[5]);
    if (pos_err < POS_TOL && rot_err < ROT_TOL) return true;

    // Jacobian
    float J[6 * DOF];
    jacobian(q, J);

    // A = J*J^T + λ^2 I (6x6)
    float A[36] = { 0 };
    for (int r = 0; r < 6; r++) {
      for (int c = 0; c < 6; c++) {
        float s = 0;
        for (int k = 0; k < DOF; k++) s += J[r * DOF + k] * J[c * DOF + k];
        A[r * 6 + c] = s;
      }
      A[r * 6 + r] += (DAMPING * DAMPING);
    }
    float invA[36];
    if (!invert6x6(A, invA)) return false;

    // y = invA * e
    float y[6];
    mat6x6_mul_vec(invA, e, y);

    // dq = J^T * y
    float dq[DOF];
    for (int i = 0; i < DOF; i++) {
      float s = 0;
      for (int r = 0; r < 6; r++) s += J[r * DOF + i] * y[r];
      dq[i] = s;
    }

    // step size limiting (simple clipping)
    const float MAX_STEP = 0.2f;  // rad per iter (or mm-equivalent via Jacobian mapping)
    for (int i = 0; i < DOF; i++) dq[i] = clampf(dq[i], -MAX_STEP, MAX_STEP);

    for (int i = 0; i < DOF; i++) q[i] += dq[i];
  }
  return false;  // not converged
}

// ---------------------------------------------------------------------------------
// Serial parsing helpers
// ---------------------------------------------------------------------------------
static bool readLine(String &out) {
  if (!Serial.available()) return false;
  out = Serial.readStringUntil('\n');
  out.trim();
  return out.length() > 0;
}

static int splitTokens(const String &s, String tokens[], int maxTok) {
  int n = 0;
  int i = 0;
  while (i < s.length() && n < maxTok) {
    while (i < s.length() && isspace(s[i])) i++;
    if (i >= s.length()) break;
    int j = i;
    while (j < s.length() && !isspace(s[j])) j++;
    tokens[n++] = s.substring(i, j);
    i = j;
  }
  return n;
}

static bool parseFloats(String tokens[], int start, int count, float *out) {
  for (int i = 0; i < count; i++) {
    char *endptr;
    String t = tokens[start + i];
    out[i] = t.toFloat();  // Arduino String has toFloat()
  }
  return true;
}

static void printArray(const float *v, int n) {
  for (int i = 0; i < n; i++) {
    if (i) Serial.print(' ');
    // print with 6 decimals
    Serial.print(v[i], 6);
  }
  Serial.println();
}

static void motormove(const float *v) {
  alocation = v[0] * radtoang;
  blocation = v[1] * radtoang;
  clocation = v[2] * radtoang;
  dlocation = v[3] * radtoang;
  elocation = v[4] * radtoang;
  flocation = v[5] * radtoang;




  if (alocation <= 90 && alocation >= -90) {
    Serial.print("A motor move = ");
    Serial.println(alocation);
    positions[0] = alocation * agear * steppower;
  }
  if (blocation <= 90 && blocation >= -90) {
    Serial.print("B motor move = ");
    Serial.println(blocation);
    positions[1] = blocation * bgear * steppower;
  }
  if (clocation <= 160 && clocation >= -160) {
    Serial.print("C motor move = ");
    Serial.println(clocation);
    positions[2] = -1 * (clocation - 5) * cgear * steppower;
  }
  if (dlocation <= 170 && dlocation >= -170) {
    Serial.print("D motor move = ");
    Serial.println(dlocation);
    positions[3] = -1 * (dlocation - 3) * dgear * steppower;
  }
  if (elocation <= 90 && elocation >= -90) {
    Serial.print("E motor move = ");
    Serial.println(elocation);
    positions[4] = elocation * egear * steppower;
  }
  if (flocation <= 180 && flocation >= -180) {
    Serial.print("F motor move = ");
    Serial.println(flocation);
    positions[5] = -1 * flocation * fgear * steppower;
  }
  robot2T.moveTo(positions);
}


void setup() {
  Serial.begin(115200);
  Serial.println("IOT robot arm ready");
  pinMode(ENABLE, OUTPUT);
  pinMode(AENABLE, OUTPUT);
  pinMode(ASTEP_PIN, OUTPUT);
  pinMode(ADIR_PIN, OUTPUT);
  pinMode(Aendstop, INPUT);
  pinMode(BSTEP_PIN, OUTPUT);
  pinMode(BDIR_PIN, OUTPUT);
  pinMode(Bendstop, INPUT);
  pinMode(CSTEP_PIN, OUTPUT);
  pinMode(CDIR_PIN, OUTPUT);
  pinMode(Cendstop, INPUT);
  pinMode(DSTEP_PIN, OUTPUT);
  pinMode(DDIR_PIN, OUTPUT);
  pinMode(Dendstop, INPUT);
  pinMode(ESTEP_PIN, OUTPUT);
  pinMode(EDIR_PIN, OUTPUT);
  pinMode(Eendstop, INPUT);
  pinMode(FSTEP_PIN, OUTPUT);
  pinMode(FDIR_PIN, OUTPUT);
  pinMode(emergbtn, INPUT);

  Wire.begin(SLAVE_ADDRESS);  // join I2C bus as a slave with address 1
  Wire.onReceive(receiveEvent); // <<< 註冊修改過的 receiveEvent 函式
  claw.attach(47);

  // Set microstepping mode (adjust as needed: HIGH or LOW)
  digitalWrite(ENABLE, LOW);
  digitalWrite(AENABLE, LOW);
  // Set the desired RPM and the max RPM


  stepperA.setMaxSpeed(maxspeed * agear);
  stepperA.setSpeed(speed * agear);
  stepperA.setAcceleration(accel * agear);

  stepperB.setMaxSpeed(maxspeed * bgear);
  stepperB.setSpeed(speed * bgear);
  stepperB.setAcceleration(accel * bgear);

  stepperC.setMaxSpeed(maxspeed * cgear);
  stepperC.setSpeed(speed * cgear);
  stepperC.setAcceleration(accel * cgear);

  stepperD.setMaxSpeed(maxspeed * dgear);
  stepperD.setSpeed(speed * dgear);
  stepperD.setAcceleration(accel * dgear);

  stepperE.setMaxSpeed(maxspeed * egear);
  stepperE.setSpeed(speed * egear);
  stepperE.setAcceleration(accel * egear);

  stepperF.setMaxSpeed(maxspeed * fgear);
  stepperF.setSpeed(speed * fgear);
  stepperF.setAcceleration(accel * fgear);

  stepperA.setPinsInverted(false, false, false);
  stepperB.setPinsInverted(true, false, false);
  stepperC.setPinsInverted(true, false, false);
  stepperD.setPinsInverted(true, false, false);
  stepperE.setPinsInverted(true, false, false);
  stepperF.setPinsInverted(true, false, false);
  claw.write(96);

  robot2T.addStepper(stepperA);
  robot2T.addStepper(stepperB);
  robot2T.addStepper(stepperC);
  robot2T.addStepper(stepperD);
  robot2T.addStepper(stepperE);
  robot2T.addStepper(stepperF);


  Serial.println(F("6DOF FK/IK ready. Units: rad & mm."));
  Serial.println(F("Commands:"));
  Serial.println(F("  FK q1 q2 q3 q4 q5 q6"));
  Serial.println(F("  IK X Y Z RX RY RZ [q1 q2 q3 q4 q5 q6]"));
}

void handleI2CCommand(); // <<< 預先宣告處理函式

void loop() {
  if (digitalRead(emergbtn) == HIGH) {
    motorzero();
  }

  // =================================================================================
  // <<< 主要修改部分 >>>
  // 檢查是否有新的 I2C 指令需要處理
  if (newDataFromI2C) {
    // 立刻重置旗標，這樣可以準備接收下一筆指令
    // 把這行放在最前面，可以避免在處理期間錯過新的指令
    newDataFromI2C = false;

    Serial.println("Received I2C command, now processing...");
    
    // 呼叫一個獨立的函式來處理指令，讓 loop() 保持乾淨
    handleI2CCommand(); 
  }
  // =================================================================================


  robot2T.run();

  String line;
  if (!readLine(line)) return;  // no blocking
  String tok[20];
  int n = splitTokens(line, tok, 20);
  if (n <= 0) return;

  if (tok[0] == "FK" && n == 1 + DOF) {
    float q[DOF];
    parseFloats(tok, 1, DOF, q);
    float T[16];
    fk(q, T);
    float rx, ry, rz;
    mat4ExtractRPY_ZYX(T, rx, ry, rz);
    float out[6] = { T[3], T[7], T[11], rx, ry, rz };
    printArray(out, 6);
  } else if (tok[0] == "IK" && (n == 1 + 6 || n == 1 + 6 + DOF)) {
    float p[3], rpy[3];
    parseFloats(tok, 1, 3, p);
    parseFloats(tok, 4, 3, rpy);
    float q[DOF] = { 0, 0, 1.57, 0, 0, 0 };
    if (n == 1 + 6 + DOF) parseFloats(tok, 1 + 6, DOF, q);

    bool ok = ik_solve(q, p, rpy);
    if (!ok) Serial.print(F("#WARN: not fully converged -> "));
    printArray(q, DOF);
    motormove(q);
  } else if (tok[0] == "HELP") {
    Serial.println(F("FK q1 q2 q3 q4 q5 q6"));
    Serial.println(F("IK X Y Z RX RY RZ [q1 q2 q3 q4 q5 q6]"));
  } else if (tok[0] == "jm") {
    alocation = tok[1].toFloat();
    blocation = tok[2].toFloat();
    clocation = tok[3].toFloat();
    dlocation = tok[4].toFloat();
    elocation = tok[5].toFloat();
    flocation = tok[6].toFloat();
    if (alocation <= 90 && alocation >= -90) {
      Serial.print("A motor move = ");
      Serial.println(alocation);
      positions[0] = alocation * agear * steppower;
      //stepperA.moveTo(alocation * agear * steppower);
    }
    if (blocation <= 90 && blocation >= -90) {
      Serial.print("B motor move = ");
      Serial.println(blocation);
      positions[1] = blocation * bgear * steppower;
      //stepperB.moveTo(blocation * bgear * steppower);
    }
    if (clocation <= 160 && clocation >= -160) {
      Serial.print("C motor move = ");
      Serial.println(clocation);
      positions[2] = -1 * (clocation - 5) * cgear * steppower;
      //stepperC.moveTo(-1 * (clocation - 5) * cgear * steppower);
    }
    if (dlocation <= 170 && dlocation >= -170) {
      Serial.print("D motor move = ");
      Serial.println(dlocation);
      positions[3] = -1 * (dlocation - 3) * dgear * steppower;
      //stepperD.moveTo(-1 * (dlocation - 3) * dgear * steppower);
    }
    if (elocation <= 90 && elocation >= -90) {
      Serial.print("E motor move = ");
      Serial.println(elocation);
      positions[4] = elocation * egear * steppower;
      //stepperE.moveTo(elocation * egear * steppower);
    }
    if (flocation <= 180 && flocation >= -180) {
      Serial.print("F motor move = ");
      Serial.println(flocation);
      positions[5] = -1 * flocation * fgear * steppower;
      //stepperF.moveTo(-1 * flocation * fgear * steppower);
    }
    robot2T.moveTo(positions);
  } else if (tok[0] == "zro") {
    motorzero();
  } else if (tok[0] == "clm") {
    clawlocation = tok[1].toFloat();
    if (clawlocation <= 180 && clawlocation >= 0) {
      Serial.print("claw motor move = ");
      Serial.println(clawlocation);
      claw.write(clawlocation);
    }
  } else {
    Serial.println(F("#ERR: bad command. Use HELP"));
  }
}


// =================================================================================
// <<< 修改後的 receiveEvent 函式 >>>
// 這個函式現在非常輕量，只負責接收字串並設定旗標，執行速度極快。
// =================================================================================
void receiveEvent(int howMany) {
  i2cCommand = ""; // 清空舊指令
  while (Wire.available() > 0)
  {
    char c = Wire.read();
    i2cCommand += c; // 將收到的字元存到全域變數中
  }
  newDataFromI2C = true; // 設定旗標，通知 loop() 有新指令
}

// =================================================================================
// <<< 新增的 I2C 指令處理函式 >>>
// 這裡面是您原本放在 receiveEvent 裡的完整處理邏輯。
// 現在它在主程式 loop() 中被安全地呼叫，不會再造成中斷問題。
// =================================================================================
void handleI2CCommand() {
  Serial.print("Full command received: ");
  Serial.println(i2cCommand); // 在這裡印出收到的完整指令

  String tok[20];
  int n = splitTokens(i2cCommand, tok, 20);
  if (n <= 0) return;

  if (tok[0] == "IK" && (n == 1 + 6)) {
    float p[3], rpy[3];
    parseFloats(tok, 1, 3, p);
    parseFloats(tok, 4, 3, rpy);
    float q[DOF] = { 0, 0, 1.57, 0, 0, 0 };
    // This part of your original code was unreachable, but I've kept it for consistency
    if (n == 1 + 6 + DOF) parseFloats(tok, 1 + 6, DOF, q); 

    bool ok = ik_solve(q, p, rpy);
    if (!ok) Serial.print(F("#WARN: not fully converged -> "));
    
    Serial.print("IK result (raw): ");
    printArray(q, DOF);
    
    // Your original rounding logic
    float qs[6];
    for (int i = 0; i < 6; i++) { // Corrected loop to process all 6 DOFs
      qs[i] = round(q[i] * 100.0) / 100.0;
    }
    Serial.print("IK result (rounded): ");
    printArray(qs, DOF);

    // Call motormove with the original high-precision values
    motormove(q);
  } 
  // You can add your other I2C commands (like "zro", "clm") here if needed
  else if (tok[0] == "zro") {
      motorzero();
  } else if (tok[0] == "clm") {
      clawlocation = tok[1].toFloat();
      if (clawlocation <= 180 && clawlocation >= 0) {
        Serial.print("claw motor move = ");
        Serial.println(clawlocation);
        claw.write(clawlocation);
      }
  } else {
      Serial.println("#ERR: Unknown I2C command or wrong parameters.");
  }
}


void motorzero() {
  stepperA.setSpeed(speed * agear);
  stepperA.setAcceleration(accel * agear);
  stepperB.setSpeed(speed * bgear);
  stepperB.setAcceleration(accel * bgear);
  stepperC.setSpeed(speed * cgear);
  stepperC.setAcceleration(accel * cgear);
  stepperD.setSpeed(speed * dgear);
  stepperD.setAcceleration(accel * dgear);
  stepperE.setSpeed(speed * egear);
  stepperE.setAcceleration(accel * egear);
  stepperF.setSpeed(speed * fgear);
  stepperF.setAcceleration(accel * fgear);
  stepperF.runToNewPosition(180 * fgear * steppower);
  stepperF.runToNewPosition(-180 * fgear * steppower);
  stepperF.runToNewPosition(0 * fgear * steppower);
  stepperE.setSpeed(-100);  //j5
  while (digitalRead(Eendstop) == LOW) {
    stepperE.runSpeed();
  }
  stepperE.setCurrentPosition(0);
  delay(200);

  stepperE.runToNewPosition(50);
  stepperE.setSpeed(-25);
  while (digitalRead(Eendstop) == LOW) {
    stepperE.runSpeed();
  }
  stepperE.setCurrentPosition(0);
  stepperE.runToNewPosition(-35);
  stepperE.setCurrentPosition(0);  //j5 zero finish

  stepperA.setSpeed(-100);  //j1
  while (digitalRead(Aendstop) == LOW) {
    stepperA.runSpeed();
  }
  stepperA.setCurrentPosition(0);
  delay(200);

  stepperA.runToNewPosition(50);
  stepperA.setSpeed(-25);
  while (digitalRead(Aendstop) == LOW) {
    stepperA.runSpeed();
  }
  stepperA.setCurrentPosition(0);  //j1 zero finish

  //stepperC.runToNewPosition(30 * cgear * steppower);  //j3 make space for j2

  stepperB.setSpeed(-100 * bgear);  //j2
  while (digitalRead(Bendstop) == LOW) {
    stepperB.runSpeed();
  }
  stepperB.setCurrentPosition(0);
  delay(200);

  stepperB.runToNewPosition(50 * bgear);
  stepperB.setSpeed(-25 * bgear);
  while (digitalRead(Bendstop) == LOW) {
    stepperB.runSpeed();
  }
  stepperB.setCurrentPosition(0);
  stepperB.runToNewPosition(-250);
  stepperB.setCurrentPosition(0);  //j2 zero finish
  delay(200);
  stepperB.runToNewPosition(90 * bgear * steppower);  //j2 make space
  stepperB.setCurrentPosition(0);

  stepperD.setSpeed(-100 * dgear);  //j4
  while (digitalRead(Dendstop) == LOW) {
    stepperD.runSpeed();
  }
  stepperD.setCurrentPosition(0);
  delay(200);

  stepperD.runToNewPosition(50 * dgear);
  stepperD.setSpeed(-25 * dgear);
  while (digitalRead(Dendstop) == LOW) {
    stepperD.runSpeed();
  }
  stepperD.setCurrentPosition(0);
  stepperD.runToNewPosition(-250);
  stepperD.setCurrentPosition(0);
  delay(200);  //j4 zero finish


  stepperC.setSpeed(-100 * cgear);
  while (digitalRead(Cendstop) == LOW) {
    stepperC.runSpeed();
  }
  stepperC.setCurrentPosition(0);
  delay(200);
  stepperC.runToNewPosition(50 * cgear);  //j3
  stepperC.setSpeed(-25 * cgear);
  while (digitalRead(Cendstop) == LOW) {
    stepperC.runSpeed();
  }
  stepperC.setCurrentPosition(0);
  stepperC.runToNewPosition(-100);
  stepperC.setCurrentPosition(0);
  stepperC.runToNewPosition(180 * cgear * steppower);
  stepperC.setCurrentPosition(0);  //j3 zero finish
  stepperE.runToNewPosition(90 * egear * steppower);
  stepperE.setCurrentPosition(0);
  stepperA.runToNewPosition(90 * agear * steppower);
  stepperA.setCurrentPosition(0);
}