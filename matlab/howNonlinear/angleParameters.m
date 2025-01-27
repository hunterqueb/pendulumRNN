
clc
clear
dt = 0.2; %sample time

t0 = 0; tf = 25;
N = 25/dt;
t = t0:dt:tf;

% Euler angles calculations
for i=1:N 
roll(i,1) = sin(3*t(i))*cos(5*t(i));
pitch(i,1) = 1.1*sin(5*t(i));
yaw(i,1) = 0.5 * (sin(5*t(i)) * (0.1+sin(3*t(i)))^3);

end
% Quternions calculations
for i=1:N
    
    qx(i,1)=sin(roll(i)/2)*cos(pitch(i)/2)*cos(yaw(i)/2)-cos(roll(i)/2)...
        *sin(pitch(i)/2)*sin(yaw(i)/2);
    qy(i,1)=cos(roll(i)/2)*sin(pitch(i)/2)*cos(yaw(i)/2)+sin(roll(i)/2)...
        *cos(pitch(i)/2)*sin(yaw(i)/2);
    qz(i,1)=cos(roll(i)/2)*cos(pitch(i)/2)*sin(yaw(i)/2)+sin(roll(i)/2)...
        *sin(pitch(i)/2)*cos(yaw(i)/2);
    qw(i,1)=cos(roll(i)/2)*cos(pitch(i)/2)*cos(yaw(i)/2)+sin(roll(i)/2)...
        *sin(pitch(i)/2)*sin(yaw(i)/2);

end
% Modified Rodrigues parameters calculations
for i=1:N
    
  p1(i,1)=qx(i)/(1+qw(i));
  p2(i,1)=qy(i)/(1+qw(i));
  p3(i,1)=qz(i)/(1+qw(i));
  
end
 
%% Plot (rodrigues parameters)
x=dt:dt:tf;

figure(1);
sgtitle('Euler Angles')
subplot(3,1,1)
plot(x,roll);
xlabel('Time (s)');
ylabel('Pitch');
subplot(3,1,2)
plot(x,pitch);
xlabel('Time (s)');
ylabel('Roll');
subplot(3,1,3)
plot(x,yaw);
xlabel('Time (s)');
ylabel('Yaw');

figure(2);
sgtitle('Modified Rodrigues Parameters')
subplot(3,1,1)
plot(x,p1);
xlabel('Time (s)');
ylabel('The first Rodrigues parameter(unitless)');
subplot(3,1,2)
plot(x,p2);
xlabel('Time (s)');
ylabel('The second Rodrigues parameter(unitless)');
subplot(3,1,3)
plot(x,p3);
xlabel('Time (s)');
ylabel('The third Rodrigues parameter(unitless)');
