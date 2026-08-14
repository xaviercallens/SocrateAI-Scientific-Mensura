/* F3 diagnostic: Benettin Lyapunov spectrum -- C workhorse.
 *
 * Re-implements, step for step, the algorithm in f3_lyapunov_ref.py:
 *   - one RK4 step on the augmented (n + n*n) vector (x, Q), so the tangent
 *     stages see the correct intermediate states;
 *   - modified Gram-Schmidt re-orthonormalisation every t_reorth time units;
 *   - exponent i = (accumulated log of column i's norm) / elapsed time.
 *
 * It exists only because the long integrations the Rossler system needs
 * (lambda_1 ~ 0.07 converges slowly, so T ~ 1e5 time units) are not tractable
 * in a NumPy loop. f3_crosscheck.py checks this program against the NumPy
 * reference on identical short runs; the NumPy reference is what the
 * known-answer battery validates. Neither is trusted on a target before both
 * of those pass.
 *
 * Also emitted, as free internal consistency checks:
 *   mean_div  time-average of trace J(x), which for ANY system must equal the
 *             sum of the exponents (the identity sum(lambda) = <div f>);
 *   mean_x    time-average of x, needed because Rossler's divergence is
 *             a + x - c, i.e. state-dependent, so its exponent sum has a
 *             measurable-but-not-constant analytic prediction.
 *
 * build: gcc -O2 -o f3_lyap f3_lyap.c -lm
 * usage: ./f3_lyap SYS DT T_TOTAL T_REORTH T_TRANSIENT X0 Y0 Z0 N_OUT
 *        SYS in {lorenz, rossler, lin3, rot, harm, vdp}
 * out:   header lines starting '#', then rows: t lam_1 .. lam_n
 *        final line: '#FINAL' t lam... mean_div mean_x
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NMAX 3

static double SIGMA = 10.0, RHO = 28.0, BETA = 8.0 / 3.0;
static double A_R = 0.2, B_R = 0.2, C_R = 5.7;
static double MU_VDP = 1.0;

/* 3x3 non-normal P with spectrum {0.5,-0.2,-1.3} conjugated in; kept
 * hard-coded (not recomputed) so the C and NumPy validation cases are the
 * same matrix to the last bit. Values printed by f3_crosscheck.py. */
static double A_LIN[9];
static double A_ROT[4] = {-0.3, -2.0, 2.0, -0.3};
static double A_HARM[4] = {0.0, 1.0, -1.0, 0.0};

typedef enum { S_LORENZ, S_ROSSLER, S_LIN3, S_ROT, S_HARM, S_VDP } sys_t;

static int sys_dim(sys_t s) {
  switch (s) {
    case S_LORENZ: case S_ROSSLER: case S_LIN3: return 3;
    default: return 2;
  }
}

static void deriv(sys_t s, const double *x, double *dx) {
  switch (s) {
    case S_LORENZ:
      dx[0] = SIGMA * (x[1] - x[0]);
      dx[1] = x[0] * (RHO - x[2]) - x[1];
      dx[2] = x[0] * x[1] - BETA * x[2];
      break;
    case S_ROSSLER:
      dx[0] = -x[1] - x[2];
      dx[1] = x[0] + A_R * x[1];
      dx[2] = B_R + x[2] * (x[0] - C_R);
      break;
    case S_LIN3:
      for (int i = 0; i < 3; i++)
        dx[i] = A_LIN[3*i] * x[0] + A_LIN[3*i+1] * x[1] + A_LIN[3*i+2] * x[2];
      break;
    case S_ROT:
      dx[0] = A_ROT[0]*x[0] + A_ROT[1]*x[1];
      dx[1] = A_ROT[2]*x[0] + A_ROT[3]*x[1];
      break;
    case S_HARM:
      dx[0] = A_HARM[0]*x[0] + A_HARM[1]*x[1];
      dx[1] = A_HARM[2]*x[0] + A_HARM[3]*x[1];
      break;
    case S_VDP:
      dx[0] = x[1];
      dx[1] = MU_VDP * (1.0 - x[0]*x[0]) * x[1] - x[0];
      break;
  }
}

/* J row-major, n x n */
static void jacobian(sys_t s, const double *x, double *J) {
  switch (s) {
    case S_LORENZ:
      J[0] = -SIGMA; J[1] = SIGMA;  J[2] = 0.0;
      J[3] = RHO - x[2]; J[4] = -1.0; J[5] = -x[0];
      J[6] = x[1]; J[7] = x[0]; J[8] = -BETA;
      break;
    case S_ROSSLER:
      J[0] = 0.0; J[1] = -1.0; J[2] = -1.0;
      J[3] = 1.0; J[4] = A_R;  J[5] = 0.0;
      J[6] = x[2]; J[7] = 0.0; J[8] = x[0] - C_R;
      break;
    case S_LIN3: memcpy(J, A_LIN, 9 * sizeof(double)); break;
    case S_ROT:  memcpy(J, A_ROT, 4 * sizeof(double)); break;
    case S_HARM: memcpy(J, A_HARM, 4 * sizeof(double)); break;
    case S_VDP:
      J[0] = 0.0; J[1] = 1.0;
      J[2] = -2.0 * MU_VDP * x[0] * x[1] - 1.0;
      J[3] = MU_VDP * (1.0 - x[0]*x[0]);
      break;
  }
}

/* augmented derivative: dx = f(x), dQ = J(x) Q  (Q row-major n x n,
 * Q[i*n+j] = component i of tangent vector j) */
static void aug_deriv(sys_t s, int n, const double *x, const double *Q,
                      double *dx, double *dQ) {
  double J[NMAX * NMAX];
  deriv(s, x, dx);
  jacobian(s, x, J);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      double acc = 0.0;
      for (int k = 0; k < n; k++) acc += J[i*n + k] * Q[k*n + j];
      dQ[i*n + j] = acc;
    }
}

static void rk4_step(sys_t s, int n, double *x, double *Q, double dt) {
  double k1x[NMAX], k2x[NMAX], k3x[NMAX], k4x[NMAX], tx[NMAX];
  double k1Q[NMAX*NMAX], k2Q[NMAX*NMAX], k3Q[NMAX*NMAX], k4Q[NMAX*NMAX], tQ[NMAX*NMAX];
  int m = n * n;

  aug_deriv(s, n, x, Q, k1x, k1Q);
  for (int i = 0; i < n; i++) tx[i] = x[i] + 0.5 * dt * k1x[i];
  for (int i = 0; i < m; i++) tQ[i] = Q[i] + 0.5 * dt * k1Q[i];
  aug_deriv(s, n, tx, tQ, k2x, k2Q);
  for (int i = 0; i < n; i++) tx[i] = x[i] + 0.5 * dt * k2x[i];
  for (int i = 0; i < m; i++) tQ[i] = Q[i] + 0.5 * dt * k2Q[i];
  aug_deriv(s, n, tx, tQ, k3x, k3Q);
  for (int i = 0; i < n; i++) tx[i] = x[i] + dt * k3x[i];
  for (int i = 0; i < m; i++) tQ[i] = Q[i] + dt * k3Q[i];
  aug_deriv(s, n, tx, tQ, k4x, k4Q);

  for (int i = 0; i < n; i++)
    x[i] += (dt / 6.0) * (k1x[i] + 2*k2x[i] + 2*k3x[i] + k4x[i]);
  for (int i = 0; i < m; i++)
    Q[i] += (dt / 6.0) * (k1Q[i] + 2*k2Q[i] + 2*k3Q[i] + k4Q[i]);
}

/* modified Gram-Schmidt on columns of Q; norms into diag */
static void mgs(int n, double *Q, double *diag) {
  for (int j = 0; j < n; j++) {
    double nrm = 0.0;
    for (int i = 0; i < n; i++) nrm += Q[i*n + j] * Q[i*n + j];
    nrm = sqrt(nrm);
    diag[j] = nrm;
    for (int i = 0; i < n; i++) Q[i*n + j] /= nrm;
    for (int c = j + 1; c < n; c++) {
      double dot = 0.0;
      for (int i = 0; i < n; i++) dot += Q[i*n + j] * Q[i*n + c];
      for (int i = 0; i < n; i++) Q[i*n + c] -= dot * Q[i*n + j];
    }
  }
}

int main(int argc, char **argv) {
  /* A_LIN = P diag(0.5,-0.2,-1.3) P^{-1}, P = [[1,2,0],[0,1,3],[2,0,1]] */
  {
    double P[9] = {1,2,0, 0,1,3, 2,0,1};
    double D[3] = {0.5, -0.2, -1.3};
    double det = 1*(1*1-3*0) - 2*(0*1-3*2) + 0*(0*0-1*2);  /* = 13 */
    double inv[9] = {
      (1*1-3*0), -(2*1-0*0),  (2*3-0*1),
     -(0*1-3*2),  (1*1-0*2), -(1*3-0*0),
      (0*0-1*2), -(1*0-2*2),  (1*1-2*0)
    };
    for (int i = 0; i < 9; i++) inv[i] /= det;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        double acc = 0.0;
        for (int k = 0; k < 3; k++) acc += P[i*3+k] * D[k] * inv[k*3+j];
        A_LIN[i*3+j] = acc;
      }
  }

  if (argc < 10) {
    fprintf(stderr, "usage: %s SYS DT T_TOTAL T_REORTH T_TRANSIENT X0 Y0 Z0 N_OUT\n", argv[0]);
    return 2;
  }
  sys_t s;
  if      (!strcmp(argv[1], "lorenz"))  s = S_LORENZ;
  else if (!strcmp(argv[1], "rossler")) s = S_ROSSLER;
  else if (!strcmp(argv[1], "lin3"))    s = S_LIN3;
  else if (!strcmp(argv[1], "rot"))     s = S_ROT;
  else if (!strcmp(argv[1], "harm"))    s = S_HARM;
  else if (!strcmp(argv[1], "vdp"))     s = S_VDP;
  else { fprintf(stderr, "unknown system %s\n", argv[1]); return 2; }

  double dt = atof(argv[2]);
  double t_total = atof(argv[3]);
  double t_reorth = atof(argv[4]);
  double t_transient = atof(argv[5]);
  double x[NMAX] = {atof(argv[6]), atof(argv[7]), atof(argv[8])};
  long n_out = atol(argv[9]);

  int n = sys_dim(s);
  double Q[NMAX*NMAX], diag[NMAX], acc[NMAX];
  for (int i = 0; i < n*n; i++) Q[i] = 0.0;
  for (int i = 0; i < n; i++) { Q[i*n+i] = 1.0; acc[i] = 0.0; }

  long steps_trans = (long)llround(t_transient / dt);
  for (long i = 0; i < steps_trans; i++) {
    double Qd[NMAX*NMAX];
    for (int a = 0; a < n*n; a++) Qd[a] = 0.0;
    for (int a = 0; a < n; a++) Qd[a*n+a] = 1.0;
    rk4_step(s, n, x, Qd, dt);
  }

  long steps_per_reorth = (long)llround(t_reorth / dt);
  if (steps_per_reorth < 1) steps_per_reorth = 1;
  long n_reorth = (long)llround(t_total / (steps_per_reorth * dt));
  long out_every = n_reorth / (n_out > 0 ? n_out : 1);
  if (out_every < 1) out_every = 1;

  double div_acc = 0.0, x_acc = 0.0;
  long n_steps = 0;

  printf("# sys=%s dt=%g t_total=%g t_reorth=%g t_transient=%g x0=(%g,%g,%g)\n",
         argv[1], dt, t_total, t_reorth, t_transient, atof(argv[6]), atof(argv[7]), atof(argv[8]));
  printf("# t");
  for (int i = 0; i < n; i++) printf(" lam%d", i+1);
  printf("\n");

  for (long r = 1; r <= n_reorth; r++) {
    for (long q = 0; q < steps_per_reorth; q++) {
      double J[NMAX*NMAX];
      jacobian(s, x, J);
      double tr = 0.0;
      for (int i = 0; i < n; i++) tr += J[i*n+i];
      div_acc += tr;
      x_acc += x[0];
      n_steps++;
      rk4_step(s, n, x, Q, dt);
    }
    mgs(n, Q, diag);
    for (int i = 0; i < n; i++) acc[i] += log(diag[i]);
    if (r % out_every == 0 || r == n_reorth) {
      double t_el = (double)r * steps_per_reorth * dt;
      printf("%.6f", t_el);
      for (int i = 0; i < n; i++) printf(" %.17g", acc[i] / t_el);
      printf("\n");
    }
  }

  double t_el = (double)n_reorth * steps_per_reorth * dt;
  printf("#FINAL %.6f", t_el);
  for (int i = 0; i < n; i++) printf(" %.17g", acc[i] / t_el);
  printf(" mean_div %.17g mean_x %.17g\n", div_acc / n_steps, x_acc / n_steps);
  return 0;
}
