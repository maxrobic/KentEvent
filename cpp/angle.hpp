#ifndef ANGLE_H
#define ANGLE_H

#include <math.h>

// Class created to handle angular data

class Angle
{
public:
 
Angle()
{
}

~Angle()
{
}

float subtract(const float a, const float b)
{
  float diff = fmod((a - b) + (float) M_PI, 2.0f * (float) M_PI) - (float) M_PI;
  return this->assure180(diff);
}

double subtract(const double a, const double b)
{
  double diff = (double) (fmod(((float) a - (float) b) + (float) M_PI, 2.0f * (float) M_PI) - (float) M_PI);
  return this->assure180(diff);
}

float radToDeg(const float a)
{
  return a * 180.0f / (float) M_PI;
}

double radToDeg(const double a)
{
  return a * 180.0f / M_PI;
}

float degToRad(const float a)
{
  return a * (float) M_PI / 180.0f;
}

double degToRad(const double a)
{
  return a * M_PI / 180.0f;
}

float assure180(const float a)
{
  float angle = fmod(a, 2.0f * (float) M_PI);
  if (angle > M_PI)
    return angle - 2.0f * (float) M_PI;
  else if (angle <= -M_PI)
    return angle + 2.0f *(float) M_PI;
  else return angle;
}

double assure180(const double a)
{
  double angle = (double) fmod((float) a, 2.0f * (float) M_PI);
  if (angle > M_PI)
    return angle - 2.0f * M_PI;
  else if (angle <= -M_PI)
    return angle + 2.0f * M_PI;
  else return angle;
}

};

#endif /* ANGLE_H */
