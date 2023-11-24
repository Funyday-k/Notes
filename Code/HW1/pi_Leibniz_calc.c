#include <stdio.h>
#include <math.h>

int main() {
  int n = 100000000;
  double pi = 0.0;

  for (int i = 0; i < n; i++) {
    pi += pow(-1, i) / (2 * i + 1);
  }

  pi *= 4;

  printf("PI = %.15lf\n", pi);

  return 0;
}