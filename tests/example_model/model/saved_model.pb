
Ø
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8â

HiddenBlock0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:D*$
shared_nameHiddenBlock0/kernel
{
'HiddenBlock0/kernel/Read/ReadVariableOpReadVariableOpHiddenBlock0/kernel*
_output_shapes

:D*
dtype0
z
HiddenBlock0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:D*"
shared_nameHiddenBlock0/bias
s
%HiddenBlock0/bias/Read/ReadVariableOpReadVariableOpHiddenBlock0/bias*
_output_shapes
:D*
dtype0

HiddenBlock1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:DB*$
shared_nameHiddenBlock1/kernel
{
'HiddenBlock1/kernel/Read/ReadVariableOpReadVariableOpHiddenBlock1/kernel*
_output_shapes

:DB*
dtype0
z
HiddenBlock1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:B*"
shared_nameHiddenBlock1/bias
s
%HiddenBlock1/bias/Read/ReadVariableOpReadVariableOpHiddenBlock1/bias*
_output_shapes
:B*
dtype0

HiddenBlock2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:BA*$
shared_nameHiddenBlock2/kernel
{
'HiddenBlock2/kernel/Read/ReadVariableOpReadVariableOpHiddenBlock2/kernel*
_output_shapes

:BA*
dtype0
z
HiddenBlock2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*"
shared_nameHiddenBlock2/bias
s
%HiddenBlock2/bias/Read/ReadVariableOpReadVariableOpHiddenBlock2/bias*
_output_shapes
:A*
dtype0

HiddenBlock3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:A@*$
shared_nameHiddenBlock3/kernel
{
'HiddenBlock3/kernel/Read/ReadVariableOpReadVariableOpHiddenBlock3/kernel*
_output_shapes

:A@*
dtype0
z
HiddenBlock3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameHiddenBlock3/bias
s
%HiddenBlock3/bias/Read/ReadVariableOpReadVariableOpHiddenBlock3/bias*
_output_shapes
:@*
dtype0
p

tau/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_name
tau/kernel
i
tau/kernel/Read/ReadVariableOpReadVariableOp
tau/kernel*
_output_shapes

:@*
dtype0
h
tau/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
tau/bias
a
tau/bias/Read/ReadVariableOpReadVariableOptau/bias*
_output_shapes
:*
dtype0
n
	dG/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_name	dG/kernel
g
dG/kernel/Read/ReadVariableOpReadVariableOp	dG/kernel*
_output_shapes

:@*
dtype0
f
dG/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	dG/bias
_
dG/bias/Read/ReadVariableOpReadVariableOpdG/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0

Adam/HiddenBlock0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:D*+
shared_nameAdam/HiddenBlock0/kernel/m

.Adam/HiddenBlock0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HiddenBlock0/kernel/m*
_output_shapes

:D*
dtype0

Adam/HiddenBlock0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:D*)
shared_nameAdam/HiddenBlock0/bias/m

,Adam/HiddenBlock0/bias/m/Read/ReadVariableOpReadVariableOpAdam/HiddenBlock0/bias/m*
_output_shapes
:D*
dtype0

Adam/HiddenBlock1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:DB*+
shared_nameAdam/HiddenBlock1/kernel/m

.Adam/HiddenBlock1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HiddenBlock1/kernel/m*
_output_shapes

:DB*
dtype0

Adam/HiddenBlock1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:B*)
shared_nameAdam/HiddenBlock1/bias/m

,Adam/HiddenBlock1/bias/m/Read/ReadVariableOpReadVariableOpAdam/HiddenBlock1/bias/m*
_output_shapes
:B*
dtype0

Adam/HiddenBlock2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:BA*+
shared_nameAdam/HiddenBlock2/kernel/m

.Adam/HiddenBlock2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HiddenBlock2/kernel/m*
_output_shapes

:BA*
dtype0

Adam/HiddenBlock2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*)
shared_nameAdam/HiddenBlock2/bias/m

,Adam/HiddenBlock2/bias/m/Read/ReadVariableOpReadVariableOpAdam/HiddenBlock2/bias/m*
_output_shapes
:A*
dtype0

Adam/HiddenBlock3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:A@*+
shared_nameAdam/HiddenBlock3/kernel/m

.Adam/HiddenBlock3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/HiddenBlock3/kernel/m*
_output_shapes

:A@*
dtype0

Adam/HiddenBlock3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/HiddenBlock3/bias/m

,Adam/HiddenBlock3/bias/m/Read/ReadVariableOpReadVariableOpAdam/HiddenBlock3/bias/m*
_output_shapes
:@*
dtype0
~
Adam/tau/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*"
shared_nameAdam/tau/kernel/m
w
%Adam/tau/kernel/m/Read/ReadVariableOpReadVariableOpAdam/tau/kernel/m*
_output_shapes

:@*
dtype0
v
Adam/tau/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/tau/bias/m
o
#Adam/tau/bias/m/Read/ReadVariableOpReadVariableOpAdam/tau/bias/m*
_output_shapes
:*
dtype0
|
Adam/dG/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_nameAdam/dG/kernel/m
u
$Adam/dG/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dG/kernel/m*
_output_shapes

:@*
dtype0
t
Adam/dG/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/dG/bias/m
m
"Adam/dG/bias/m/Read/ReadVariableOpReadVariableOpAdam/dG/bias/m*
_output_shapes
:*
dtype0

Adam/HiddenBlock0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:D*+
shared_nameAdam/HiddenBlock0/kernel/v

.Adam/HiddenBlock0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HiddenBlock0/kernel/v*
_output_shapes

:D*
dtype0

Adam/HiddenBlock0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:D*)
shared_nameAdam/HiddenBlock0/bias/v

,Adam/HiddenBlock0/bias/v/Read/ReadVariableOpReadVariableOpAdam/HiddenBlock0/bias/v*
_output_shapes
:D*
dtype0

Adam/HiddenBlock1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:DB*+
shared_nameAdam/HiddenBlock1/kernel/v

.Adam/HiddenBlock1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HiddenBlock1/kernel/v*
_output_shapes

:DB*
dtype0

Adam/HiddenBlock1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:B*)
shared_nameAdam/HiddenBlock1/bias/v

,Adam/HiddenBlock1/bias/v/Read/ReadVariableOpReadVariableOpAdam/HiddenBlock1/bias/v*
_output_shapes
:B*
dtype0

Adam/HiddenBlock2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:BA*+
shared_nameAdam/HiddenBlock2/kernel/v

.Adam/HiddenBlock2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HiddenBlock2/kernel/v*
_output_shapes

:BA*
dtype0

Adam/HiddenBlock2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*)
shared_nameAdam/HiddenBlock2/bias/v

,Adam/HiddenBlock2/bias/v/Read/ReadVariableOpReadVariableOpAdam/HiddenBlock2/bias/v*
_output_shapes
:A*
dtype0

Adam/HiddenBlock3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:A@*+
shared_nameAdam/HiddenBlock3/kernel/v

.Adam/HiddenBlock3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/HiddenBlock3/kernel/v*
_output_shapes

:A@*
dtype0

Adam/HiddenBlock3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/HiddenBlock3/bias/v

,Adam/HiddenBlock3/bias/v/Read/ReadVariableOpReadVariableOpAdam/HiddenBlock3/bias/v*
_output_shapes
:@*
dtype0
~
Adam/tau/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*"
shared_nameAdam/tau/kernel/v
w
%Adam/tau/kernel/v/Read/ReadVariableOpReadVariableOpAdam/tau/kernel/v*
_output_shapes

:@*
dtype0
v
Adam/tau/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/tau/bias/v
o
#Adam/tau/bias/v/Read/ReadVariableOpReadVariableOpAdam/tau/bias/v*
_output_shapes
:*
dtype0
|
Adam/dG/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_nameAdam/dG/kernel/v
u
$Adam/dG/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dG/kernel/v*
_output_shapes

:@*
dtype0
t
Adam/dG/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/dG/bias/v
m
"Adam/dG/bias/v/Read/ReadVariableOpReadVariableOpAdam/dG/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
G
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÁF
value·FB´F B­F

layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
	optimizer
loss
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
 
 
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
h

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
h

(kernel
)bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
h

.kernel
/bias
0trainable_variables
1regularization_losses
2	variables
3	keras_api
h

4kernel
5bias
6trainable_variables
7regularization_losses
8	variables
9	keras_api
¬
:iter

;beta_1

<beta_2
	=decay
>learning_ratem|m}m~m"m#m(m)m.m/m4m5mvvvv"v#v(v)v.v/v4v5v
 
V
0
1
2
3
"4
#5
(6
)7
.8
/9
410
511
 
V
0
1
2
3
"4
#5
(6
)7
.8
/9
410
511
­
trainable_variables
regularization_losses
?metrics
@layer_metrics

Alayers
	variables
Bnon_trainable_variables
Clayer_regularization_losses
 
 
 
 
­
trainable_variables
Dlayer_metrics
Emetrics
regularization_losses

Flayers
	variables
Gnon_trainable_variables
Hlayer_regularization_losses
_]
VARIABLE_VALUEHiddenBlock0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEHiddenBlock0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
trainable_variables
Ilayer_metrics
Jmetrics
regularization_losses

Klayers
	variables
Lnon_trainable_variables
Mlayer_regularization_losses
_]
VARIABLE_VALUEHiddenBlock1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEHiddenBlock1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
trainable_variables
Nlayer_metrics
Ometrics
regularization_losses

Players
 	variables
Qnon_trainable_variables
Rlayer_regularization_losses
_]
VARIABLE_VALUEHiddenBlock2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEHiddenBlock2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
­
$trainable_variables
Slayer_metrics
Tmetrics
%regularization_losses

Ulayers
&	variables
Vnon_trainable_variables
Wlayer_regularization_losses
_]
VARIABLE_VALUEHiddenBlock3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEHiddenBlock3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
­
*trainable_variables
Xlayer_metrics
Ymetrics
+regularization_losses

Zlayers
,	variables
[non_trainable_variables
\layer_regularization_losses
VT
VARIABLE_VALUE
tau/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEtau/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
 

.0
/1
­
0trainable_variables
]layer_metrics
^metrics
1regularization_losses

_layers
2	variables
`non_trainable_variables
alayer_regularization_losses
US
VARIABLE_VALUE	dG/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEdG/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51
 

40
51
­
6trainable_variables
blayer_metrics
cmetrics
7regularization_losses

dlayers
8	variables
enon_trainable_variables
flayer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

g0
h1
i2
j3
 
F
0
1
2
3
4
5
6
7
	8

9
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ktotal
	lcount
m	variables
n	keras_api
4
	ototal
	pcount
q	variables
r	keras_api
4
	stotal
	tcount
u	variables
v	keras_api
D
	wtotal
	xcount
y
_fn_kwargs
z	variables
{	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

k0
l1

m	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

o0
p1

q	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

s0
t1

u	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE
 

w0
x1

z	variables

VARIABLE_VALUEAdam/HiddenBlock0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/HiddenBlock0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/HiddenBlock1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/HiddenBlock1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/HiddenBlock2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/HiddenBlock2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/HiddenBlock3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/HiddenBlock3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/tau/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/tau/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dG/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/dG/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/HiddenBlock0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/HiddenBlock0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/HiddenBlock1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/HiddenBlock1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/HiddenBlock2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/HiddenBlock2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/HiddenBlock3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/HiddenBlock3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/tau/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/tau/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dG/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/dG/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
t
serving_default_FPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
t
serving_default_TPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
t
serving_default_nPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Ê
StatefulPartitionedCallStatefulPartitionedCallserving_default_Fserving_default_Tserving_default_nHiddenBlock0/kernelHiddenBlock0/biasHiddenBlock1/kernelHiddenBlock1/biasHiddenBlock2/kernelHiddenBlock2/biasHiddenBlock3/kernelHiddenBlock3/bias	dG/kerneldG/bias
tau/kerneltau/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_126871
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¯
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'HiddenBlock0/kernel/Read/ReadVariableOp%HiddenBlock0/bias/Read/ReadVariableOp'HiddenBlock1/kernel/Read/ReadVariableOp%HiddenBlock1/bias/Read/ReadVariableOp'HiddenBlock2/kernel/Read/ReadVariableOp%HiddenBlock2/bias/Read/ReadVariableOp'HiddenBlock3/kernel/Read/ReadVariableOp%HiddenBlock3/bias/Read/ReadVariableOptau/kernel/Read/ReadVariableOptau/bias/Read/ReadVariableOpdG/kernel/Read/ReadVariableOpdG/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOp.Adam/HiddenBlock0/kernel/m/Read/ReadVariableOp,Adam/HiddenBlock0/bias/m/Read/ReadVariableOp.Adam/HiddenBlock1/kernel/m/Read/ReadVariableOp,Adam/HiddenBlock1/bias/m/Read/ReadVariableOp.Adam/HiddenBlock2/kernel/m/Read/ReadVariableOp,Adam/HiddenBlock2/bias/m/Read/ReadVariableOp.Adam/HiddenBlock3/kernel/m/Read/ReadVariableOp,Adam/HiddenBlock3/bias/m/Read/ReadVariableOp%Adam/tau/kernel/m/Read/ReadVariableOp#Adam/tau/bias/m/Read/ReadVariableOp$Adam/dG/kernel/m/Read/ReadVariableOp"Adam/dG/bias/m/Read/ReadVariableOp.Adam/HiddenBlock0/kernel/v/Read/ReadVariableOp,Adam/HiddenBlock0/bias/v/Read/ReadVariableOp.Adam/HiddenBlock1/kernel/v/Read/ReadVariableOp,Adam/HiddenBlock1/bias/v/Read/ReadVariableOp.Adam/HiddenBlock2/kernel/v/Read/ReadVariableOp,Adam/HiddenBlock2/bias/v/Read/ReadVariableOp.Adam/HiddenBlock3/kernel/v/Read/ReadVariableOp,Adam/HiddenBlock3/bias/v/Read/ReadVariableOp%Adam/tau/kernel/v/Read/ReadVariableOp#Adam/tau/bias/v/Read/ReadVariableOp$Adam/dG/kernel/v/Read/ReadVariableOp"Adam/dG/bias/v/Read/ReadVariableOpConst*>
Tin7
523	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_127341
Ö	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameHiddenBlock0/kernelHiddenBlock0/biasHiddenBlock1/kernelHiddenBlock1/biasHiddenBlock2/kernelHiddenBlock2/biasHiddenBlock3/kernelHiddenBlock3/bias
tau/kerneltau/bias	dG/kerneldG/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2total_3count_3Adam/HiddenBlock0/kernel/mAdam/HiddenBlock0/bias/mAdam/HiddenBlock1/kernel/mAdam/HiddenBlock1/bias/mAdam/HiddenBlock2/kernel/mAdam/HiddenBlock2/bias/mAdam/HiddenBlock3/kernel/mAdam/HiddenBlock3/bias/mAdam/tau/kernel/mAdam/tau/bias/mAdam/dG/kernel/mAdam/dG/bias/mAdam/HiddenBlock0/kernel/vAdam/HiddenBlock0/bias/vAdam/HiddenBlock1/kernel/vAdam/HiddenBlock1/bias/vAdam/HiddenBlock2/kernel/vAdam/HiddenBlock2/bias/vAdam/HiddenBlock3/kernel/vAdam/HiddenBlock3/bias/vAdam/tau/kernel/vAdam/tau/bias/vAdam/dG/kernel/vAdam/dG/bias/v*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_127498¼
Î
µ
0__inference_collision_model_layer_call_fn_126828
n
t
f
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallntfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_collision_model_layer_call_and_return_conditional_losses_1267992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namen:JF
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameT:JF
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameF
	
Ø
?__inference_tau_layer_call_and_return_conditional_losses_126629

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

©
$__inference_signature_wrapper_126871
f
t
n
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallntfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_1264612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameF:JF
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameT:JF
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namen
¿(
à
K__inference_collision_model_layer_call_and_return_conditional_losses_126685
n
t
f
hiddenblock0_126653
hiddenblock0_126655
hiddenblock1_126658
hiddenblock1_126660
hiddenblock2_126663
hiddenblock2_126665
hiddenblock3_126668
hiddenblock3_126670
	dg_126673
	dg_126675

tau_126678

tau_126680
identity

identity_1¢$HiddenBlock0/StatefulPartitionedCall¢$HiddenBlock1/StatefulPartitionedCall¢$HiddenBlock2/StatefulPartitionedCall¢$HiddenBlock3/StatefulPartitionedCall¢dG/StatefulPartitionedCall¢tau/StatefulPartitionedCallÝ
InputBlock/PartitionedCallPartitionedCallntf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_InputBlock_layer_call_and_return_conditional_losses_1264752
InputBlock/PartitionedCallÅ
$HiddenBlock0/StatefulPartitionedCallStatefulPartitionedCall#InputBlock/PartitionedCall:output:0hiddenblock0_126653hiddenblock0_126655*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock0_layer_call_and_return_conditional_losses_1264962&
$HiddenBlock0/StatefulPartitionedCallÏ
$HiddenBlock1/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock0/StatefulPartitionedCall:output:0hiddenblock1_126658hiddenblock1_126660*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock1_layer_call_and_return_conditional_losses_1265232&
$HiddenBlock1/StatefulPartitionedCallÏ
$HiddenBlock2/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock1/StatefulPartitionedCall:output:0hiddenblock2_126663hiddenblock2_126665*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock2_layer_call_and_return_conditional_losses_1265502&
$HiddenBlock2/StatefulPartitionedCallÏ
$HiddenBlock3/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock2/StatefulPartitionedCall:output:0hiddenblock3_126668hiddenblock3_126670*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock3_layer_call_and_return_conditional_losses_1265772&
$HiddenBlock3/StatefulPartitionedCall
dG/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock3/StatefulPartitionedCall:output:0	dg_126673	dg_126675*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_dG_layer_call_and_return_conditional_losses_1266032
dG/StatefulPartitionedCall¢
tau/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock3/StatefulPartitionedCall:output:0
tau_126678
tau_126680*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_tau_layer_call_and_return_conditional_losses_1266292
tau/StatefulPartitionedCallÏ
IdentityIdentity$tau/StatefulPartitionedCall:output:0%^HiddenBlock0/StatefulPartitionedCall%^HiddenBlock1/StatefulPartitionedCall%^HiddenBlock2/StatefulPartitionedCall%^HiddenBlock3/StatefulPartitionedCall^dG/StatefulPartitionedCall^tau/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÒ

Identity_1Identity#dG/StatefulPartitionedCall:output:0%^HiddenBlock0/StatefulPartitionedCall%^HiddenBlock1/StatefulPartitionedCall%^HiddenBlock2/StatefulPartitionedCall%^HiddenBlock3/StatefulPartitionedCall^dG/StatefulPartitionedCall^tau/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::2L
$HiddenBlock0/StatefulPartitionedCall$HiddenBlock0/StatefulPartitionedCall2L
$HiddenBlock1/StatefulPartitionedCall$HiddenBlock1/StatefulPartitionedCall2L
$HiddenBlock2/StatefulPartitionedCall$HiddenBlock2/StatefulPartitionedCall2L
$HiddenBlock3/StatefulPartitionedCall$HiddenBlock3/StatefulPartitionedCall28
dG/StatefulPartitionedCalldG/StatefulPartitionedCall2:
tau/StatefulPartitionedCalltau/StatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namen:JF
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameT:JF
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameF

e
+__inference_InputBlock_layer_call_fn_127050
inputs_0
inputs_1
inputs_2
identityÜ
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_InputBlock_layer_call_and_return_conditional_losses_1264752
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2
þÌ

"__inference__traced_restore_127498
file_prefix(
$assignvariableop_hiddenblock0_kernel(
$assignvariableop_1_hiddenblock0_bias*
&assignvariableop_2_hiddenblock1_kernel(
$assignvariableop_3_hiddenblock1_bias*
&assignvariableop_4_hiddenblock2_kernel(
$assignvariableop_5_hiddenblock2_bias*
&assignvariableop_6_hiddenblock3_kernel(
$assignvariableop_7_hiddenblock3_bias!
assignvariableop_8_tau_kernel
assignvariableop_9_tau_bias!
assignvariableop_10_dg_kernel
assignvariableop_11_dg_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_1
assignvariableop_21_total_2
assignvariableop_22_count_2
assignvariableop_23_total_3
assignvariableop_24_count_32
.assignvariableop_25_adam_hiddenblock0_kernel_m0
,assignvariableop_26_adam_hiddenblock0_bias_m2
.assignvariableop_27_adam_hiddenblock1_kernel_m0
,assignvariableop_28_adam_hiddenblock1_bias_m2
.assignvariableop_29_adam_hiddenblock2_kernel_m0
,assignvariableop_30_adam_hiddenblock2_bias_m2
.assignvariableop_31_adam_hiddenblock3_kernel_m0
,assignvariableop_32_adam_hiddenblock3_bias_m)
%assignvariableop_33_adam_tau_kernel_m'
#assignvariableop_34_adam_tau_bias_m(
$assignvariableop_35_adam_dg_kernel_m&
"assignvariableop_36_adam_dg_bias_m2
.assignvariableop_37_adam_hiddenblock0_kernel_v0
,assignvariableop_38_adam_hiddenblock0_bias_v2
.assignvariableop_39_adam_hiddenblock1_kernel_v0
,assignvariableop_40_adam_hiddenblock1_bias_v2
.assignvariableop_41_adam_hiddenblock2_kernel_v0
,assignvariableop_42_adam_hiddenblock2_bias_v2
.assignvariableop_43_adam_hiddenblock3_kernel_v0
,assignvariableop_44_adam_hiddenblock3_bias_v)
%assignvariableop_45_adam_tau_kernel_v'
#assignvariableop_46_adam_tau_bias_v(
$assignvariableop_47_adam_dg_kernel_v&
"assignvariableop_48_adam_dg_bias_v
identity_50¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*¤
valueB2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesò
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices¨
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Þ
_output_shapesË
È::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity£
AssignVariableOpAssignVariableOp$assignvariableop_hiddenblock0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1©
AssignVariableOp_1AssignVariableOp$assignvariableop_1_hiddenblock0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2«
AssignVariableOp_2AssignVariableOp&assignvariableop_2_hiddenblock1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3©
AssignVariableOp_3AssignVariableOp$assignvariableop_3_hiddenblock1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4«
AssignVariableOp_4AssignVariableOp&assignvariableop_4_hiddenblock2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5©
AssignVariableOp_5AssignVariableOp$assignvariableop_5_hiddenblock2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6«
AssignVariableOp_6AssignVariableOp&assignvariableop_6_hiddenblock3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7©
AssignVariableOp_7AssignVariableOp$assignvariableop_7_hiddenblock3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¢
AssignVariableOp_8AssignVariableOpassignvariableop_8_tau_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9 
AssignVariableOp_9AssignVariableOpassignvariableop_9_tau_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¥
AssignVariableOp_10AssignVariableOpassignvariableop_10_dg_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11£
AssignVariableOp_11AssignVariableOpassignvariableop_11_dg_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12¥
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13§
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14§
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¦
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16®
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¡
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¡
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19£
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20£
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21£
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_2Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22£
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23£
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_3Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24£
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_3Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¶
AssignVariableOp_25AssignVariableOp.assignvariableop_25_adam_hiddenblock0_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26´
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_hiddenblock0_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¶
AssignVariableOp_27AssignVariableOp.assignvariableop_27_adam_hiddenblock1_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28´
AssignVariableOp_28AssignVariableOp,assignvariableop_28_adam_hiddenblock1_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¶
AssignVariableOp_29AssignVariableOp.assignvariableop_29_adam_hiddenblock2_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30´
AssignVariableOp_30AssignVariableOp,assignvariableop_30_adam_hiddenblock2_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31¶
AssignVariableOp_31AssignVariableOp.assignvariableop_31_adam_hiddenblock3_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32´
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_hiddenblock3_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33­
AssignVariableOp_33AssignVariableOp%assignvariableop_33_adam_tau_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34«
AssignVariableOp_34AssignVariableOp#assignvariableop_34_adam_tau_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¬
AssignVariableOp_35AssignVariableOp$assignvariableop_35_adam_dg_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36ª
AssignVariableOp_36AssignVariableOp"assignvariableop_36_adam_dg_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37¶
AssignVariableOp_37AssignVariableOp.assignvariableop_37_adam_hiddenblock0_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38´
AssignVariableOp_38AssignVariableOp,assignvariableop_38_adam_hiddenblock0_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¶
AssignVariableOp_39AssignVariableOp.assignvariableop_39_adam_hiddenblock1_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40´
AssignVariableOp_40AssignVariableOp,assignvariableop_40_adam_hiddenblock1_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41¶
AssignVariableOp_41AssignVariableOp.assignvariableop_41_adam_hiddenblock2_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42´
AssignVariableOp_42AssignVariableOp,assignvariableop_42_adam_hiddenblock2_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43¶
AssignVariableOp_43AssignVariableOp.assignvariableop_43_adam_hiddenblock3_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44´
AssignVariableOp_44AssignVariableOp,assignvariableop_44_adam_hiddenblock3_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45­
AssignVariableOp_45AssignVariableOp%assignvariableop_45_adam_tau_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46«
AssignVariableOp_46AssignVariableOp#assignvariableop_46_adam_tau_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47¬
AssignVariableOp_47AssignVariableOp$assignvariableop_47_adam_dg_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48ª
AssignVariableOp_48AssignVariableOp"assignvariableop_48_adam_dg_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_489
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_49	
Identity_50IdentityIdentity_49:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_50"#
identity_50Identity_50:output:0*Û
_input_shapesÉ
Æ: :::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ò
y
$__inference_tau_layer_call_fn_127149

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_tau_layer_call_and_return_conditional_losses_1266292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ô	
á
H__inference_HiddenBlock3_layer_call_and_return_conditional_losses_126577

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:A@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
	
×
>__inference_dG_layer_call_and_return_conditional_losses_126603

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ô	
á
H__inference_HiddenBlock0_layer_call_and_return_conditional_losses_127061

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:D*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:D*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å

-__inference_HiddenBlock0_layer_call_fn_127070

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock0_layer_call_and_return_conditional_losses_1264962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô(
ó
K__inference_collision_model_layer_call_and_return_conditional_losses_126799

inputs
inputs_1
inputs_2
hiddenblock0_126767
hiddenblock0_126769
hiddenblock1_126772
hiddenblock1_126774
hiddenblock2_126777
hiddenblock2_126779
hiddenblock3_126782
hiddenblock3_126784
	dg_126787
	dg_126789

tau_126792

tau_126794
identity

identity_1¢$HiddenBlock0/StatefulPartitionedCall¢$HiddenBlock1/StatefulPartitionedCall¢$HiddenBlock2/StatefulPartitionedCall¢$HiddenBlock3/StatefulPartitionedCall¢dG/StatefulPartitionedCall¢tau/StatefulPartitionedCallð
InputBlock/PartitionedCallPartitionedCallinputsinputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_InputBlock_layer_call_and_return_conditional_losses_1264752
InputBlock/PartitionedCallÅ
$HiddenBlock0/StatefulPartitionedCallStatefulPartitionedCall#InputBlock/PartitionedCall:output:0hiddenblock0_126767hiddenblock0_126769*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock0_layer_call_and_return_conditional_losses_1264962&
$HiddenBlock0/StatefulPartitionedCallÏ
$HiddenBlock1/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock0/StatefulPartitionedCall:output:0hiddenblock1_126772hiddenblock1_126774*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock1_layer_call_and_return_conditional_losses_1265232&
$HiddenBlock1/StatefulPartitionedCallÏ
$HiddenBlock2/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock1/StatefulPartitionedCall:output:0hiddenblock2_126777hiddenblock2_126779*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock2_layer_call_and_return_conditional_losses_1265502&
$HiddenBlock2/StatefulPartitionedCallÏ
$HiddenBlock3/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock2/StatefulPartitionedCall:output:0hiddenblock3_126782hiddenblock3_126784*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock3_layer_call_and_return_conditional_losses_1265772&
$HiddenBlock3/StatefulPartitionedCall
dG/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock3/StatefulPartitionedCall:output:0	dg_126787	dg_126789*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_dG_layer_call_and_return_conditional_losses_1266032
dG/StatefulPartitionedCall¢
tau/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock3/StatefulPartitionedCall:output:0
tau_126792
tau_126794*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_tau_layer_call_and_return_conditional_losses_1266292
tau/StatefulPartitionedCallÏ
IdentityIdentity$tau/StatefulPartitionedCall:output:0%^HiddenBlock0/StatefulPartitionedCall%^HiddenBlock1/StatefulPartitionedCall%^HiddenBlock2/StatefulPartitionedCall%^HiddenBlock3/StatefulPartitionedCall^dG/StatefulPartitionedCall^tau/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÒ

Identity_1Identity#dG/StatefulPartitionedCall:output:0%^HiddenBlock0/StatefulPartitionedCall%^HiddenBlock1/StatefulPartitionedCall%^HiddenBlock2/StatefulPartitionedCall%^HiddenBlock3/StatefulPartitionedCall^dG/StatefulPartitionedCall^tau/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::2L
$HiddenBlock0/StatefulPartitionedCall$HiddenBlock0/StatefulPartitionedCall2L
$HiddenBlock1/StatefulPartitionedCall$HiddenBlock1/StatefulPartitionedCall2L
$HiddenBlock2/StatefulPartitionedCall$HiddenBlock2/StatefulPartitionedCall2L
$HiddenBlock3/StatefulPartitionedCall$HiddenBlock3/StatefulPartitionedCall28
dG/StatefulPartitionedCalldG/StatefulPartitionedCall2:
tau/StatefulPartitionedCalltau/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô	
á
H__inference_HiddenBlock3_layer_call_and_return_conditional_losses_127121

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:A@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
ô(
ó
K__inference_collision_model_layer_call_and_return_conditional_losses_126728

inputs
inputs_1
inputs_2
hiddenblock0_126696
hiddenblock0_126698
hiddenblock1_126701
hiddenblock1_126703
hiddenblock2_126706
hiddenblock2_126708
hiddenblock3_126711
hiddenblock3_126713
	dg_126716
	dg_126718

tau_126721

tau_126723
identity

identity_1¢$HiddenBlock0/StatefulPartitionedCall¢$HiddenBlock1/StatefulPartitionedCall¢$HiddenBlock2/StatefulPartitionedCall¢$HiddenBlock3/StatefulPartitionedCall¢dG/StatefulPartitionedCall¢tau/StatefulPartitionedCallð
InputBlock/PartitionedCallPartitionedCallinputsinputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_InputBlock_layer_call_and_return_conditional_losses_1264752
InputBlock/PartitionedCallÅ
$HiddenBlock0/StatefulPartitionedCallStatefulPartitionedCall#InputBlock/PartitionedCall:output:0hiddenblock0_126696hiddenblock0_126698*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock0_layer_call_and_return_conditional_losses_1264962&
$HiddenBlock0/StatefulPartitionedCallÏ
$HiddenBlock1/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock0/StatefulPartitionedCall:output:0hiddenblock1_126701hiddenblock1_126703*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock1_layer_call_and_return_conditional_losses_1265232&
$HiddenBlock1/StatefulPartitionedCallÏ
$HiddenBlock2/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock1/StatefulPartitionedCall:output:0hiddenblock2_126706hiddenblock2_126708*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock2_layer_call_and_return_conditional_losses_1265502&
$HiddenBlock2/StatefulPartitionedCallÏ
$HiddenBlock3/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock2/StatefulPartitionedCall:output:0hiddenblock3_126711hiddenblock3_126713*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock3_layer_call_and_return_conditional_losses_1265772&
$HiddenBlock3/StatefulPartitionedCall
dG/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock3/StatefulPartitionedCall:output:0	dg_126716	dg_126718*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_dG_layer_call_and_return_conditional_losses_1266032
dG/StatefulPartitionedCall¢
tau/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock3/StatefulPartitionedCall:output:0
tau_126721
tau_126723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_tau_layer_call_and_return_conditional_losses_1266292
tau/StatefulPartitionedCallÏ
IdentityIdentity$tau/StatefulPartitionedCall:output:0%^HiddenBlock0/StatefulPartitionedCall%^HiddenBlock1/StatefulPartitionedCall%^HiddenBlock2/StatefulPartitionedCall%^HiddenBlock3/StatefulPartitionedCall^dG/StatefulPartitionedCall^tau/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÒ

Identity_1Identity#dG/StatefulPartitionedCall:output:0%^HiddenBlock0/StatefulPartitionedCall%^HiddenBlock1/StatefulPartitionedCall%^HiddenBlock2/StatefulPartitionedCall%^HiddenBlock3/StatefulPartitionedCall^dG/StatefulPartitionedCall^tau/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::2L
$HiddenBlock0/StatefulPartitionedCall$HiddenBlock0/StatefulPartitionedCall2L
$HiddenBlock1/StatefulPartitionedCall$HiddenBlock1/StatefulPartitionedCall2L
$HiddenBlock2/StatefulPartitionedCall$HiddenBlock2/StatefulPartitionedCall2L
$HiddenBlock3/StatefulPartitionedCall$HiddenBlock3/StatefulPartitionedCall28
dG/StatefulPartitionedCalldG/StatefulPartitionedCall2:
tau/StatefulPartitionedCalltau/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯
~
F__inference_InputBlock_layer_call_and_return_conditional_losses_126475

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å

-__inference_HiddenBlock2_layer_call_fn_127110

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock2_layer_call_and_return_conditional_losses_1265502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿB::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB
 
_user_specified_nameinputs
¿(
à
K__inference_collision_model_layer_call_and_return_conditional_losses_126647
n
t
f
hiddenblock0_126507
hiddenblock0_126509
hiddenblock1_126534
hiddenblock1_126536
hiddenblock2_126561
hiddenblock2_126563
hiddenblock3_126588
hiddenblock3_126590
	dg_126614
	dg_126616

tau_126640

tau_126642
identity

identity_1¢$HiddenBlock0/StatefulPartitionedCall¢$HiddenBlock1/StatefulPartitionedCall¢$HiddenBlock2/StatefulPartitionedCall¢$HiddenBlock3/StatefulPartitionedCall¢dG/StatefulPartitionedCall¢tau/StatefulPartitionedCallÝ
InputBlock/PartitionedCallPartitionedCallntf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_InputBlock_layer_call_and_return_conditional_losses_1264752
InputBlock/PartitionedCallÅ
$HiddenBlock0/StatefulPartitionedCallStatefulPartitionedCall#InputBlock/PartitionedCall:output:0hiddenblock0_126507hiddenblock0_126509*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock0_layer_call_and_return_conditional_losses_1264962&
$HiddenBlock0/StatefulPartitionedCallÏ
$HiddenBlock1/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock0/StatefulPartitionedCall:output:0hiddenblock1_126534hiddenblock1_126536*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock1_layer_call_and_return_conditional_losses_1265232&
$HiddenBlock1/StatefulPartitionedCallÏ
$HiddenBlock2/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock1/StatefulPartitionedCall:output:0hiddenblock2_126561hiddenblock2_126563*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock2_layer_call_and_return_conditional_losses_1265502&
$HiddenBlock2/StatefulPartitionedCallÏ
$HiddenBlock3/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock2/StatefulPartitionedCall:output:0hiddenblock3_126588hiddenblock3_126590*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock3_layer_call_and_return_conditional_losses_1265772&
$HiddenBlock3/StatefulPartitionedCall
dG/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock3/StatefulPartitionedCall:output:0	dg_126614	dg_126616*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_dG_layer_call_and_return_conditional_losses_1266032
dG/StatefulPartitionedCall¢
tau/StatefulPartitionedCallStatefulPartitionedCall-HiddenBlock3/StatefulPartitionedCall:output:0
tau_126640
tau_126642*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_tau_layer_call_and_return_conditional_losses_1266292
tau/StatefulPartitionedCallÏ
IdentityIdentity$tau/StatefulPartitionedCall:output:0%^HiddenBlock0/StatefulPartitionedCall%^HiddenBlock1/StatefulPartitionedCall%^HiddenBlock2/StatefulPartitionedCall%^HiddenBlock3/StatefulPartitionedCall^dG/StatefulPartitionedCall^tau/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÒ

Identity_1Identity#dG/StatefulPartitionedCall:output:0%^HiddenBlock0/StatefulPartitionedCall%^HiddenBlock1/StatefulPartitionedCall%^HiddenBlock2/StatefulPartitionedCall%^HiddenBlock3/StatefulPartitionedCall^dG/StatefulPartitionedCall^tau/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::2L
$HiddenBlock0/StatefulPartitionedCall$HiddenBlock0/StatefulPartitionedCall2L
$HiddenBlock1/StatefulPartitionedCall$HiddenBlock1/StatefulPartitionedCall2L
$HiddenBlock2/StatefulPartitionedCall$HiddenBlock2/StatefulPartitionedCall2L
$HiddenBlock3/StatefulPartitionedCall$HiddenBlock3/StatefulPartitionedCall28
dG/StatefulPartitionedCalldG/StatefulPartitionedCall2:
tau/StatefulPartitionedCalltau/StatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namen:JF
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameT:JF
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameF
Î
µ
0__inference_collision_model_layer_call_fn_126757
n
t
f
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallntfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_collision_model_layer_call_and_return_conditional_losses_1267282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namen:JF
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameT:JF
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameF
ô	
á
H__inference_HiddenBlock1_layer_call_and_return_conditional_losses_126523

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:DB*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:B*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿD::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
 
_user_specified_nameinputs

Ê
0__inference_collision_model_layer_call_fn_127035
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_collision_model_layer_call_and_return_conditional_losses_1267992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2
¿V
¡
!__inference__wrapped_model_126461
n
t
f?
;collision_model_hiddenblock0_matmul_readvariableop_resource@
<collision_model_hiddenblock0_biasadd_readvariableop_resource?
;collision_model_hiddenblock1_matmul_readvariableop_resource@
<collision_model_hiddenblock1_biasadd_readvariableop_resource?
;collision_model_hiddenblock2_matmul_readvariableop_resource@
<collision_model_hiddenblock2_biasadd_readvariableop_resource?
;collision_model_hiddenblock3_matmul_readvariableop_resource@
<collision_model_hiddenblock3_biasadd_readvariableop_resource5
1collision_model_dg_matmul_readvariableop_resource6
2collision_model_dg_biasadd_readvariableop_resource6
2collision_model_tau_matmul_readvariableop_resource7
3collision_model_tau_biasadd_readvariableop_resource
identity

identity_1¢3collision_model/HiddenBlock0/BiasAdd/ReadVariableOp¢2collision_model/HiddenBlock0/MatMul/ReadVariableOp¢3collision_model/HiddenBlock1/BiasAdd/ReadVariableOp¢2collision_model/HiddenBlock1/MatMul/ReadVariableOp¢3collision_model/HiddenBlock2/BiasAdd/ReadVariableOp¢2collision_model/HiddenBlock2/MatMul/ReadVariableOp¢3collision_model/HiddenBlock3/BiasAdd/ReadVariableOp¢2collision_model/HiddenBlock3/MatMul/ReadVariableOp¢)collision_model/dG/BiasAdd/ReadVariableOp¢(collision_model/dG/MatMul/ReadVariableOp¢*collision_model/tau/BiasAdd/ReadVariableOp¢)collision_model/tau/MatMul/ReadVariableOp
&collision_model/InputBlock/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2(
&collision_model/InputBlock/concat/axisÇ
!collision_model/InputBlock/concatConcatV2ntf/collision_model/InputBlock/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!collision_model/InputBlock/concatä
2collision_model/HiddenBlock0/MatMul/ReadVariableOpReadVariableOp;collision_model_hiddenblock0_matmul_readvariableop_resource*
_output_shapes

:D*
dtype024
2collision_model/HiddenBlock0/MatMul/ReadVariableOpî
#collision_model/HiddenBlock0/MatMulMatMul*collision_model/InputBlock/concat:output:0:collision_model/HiddenBlock0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD2%
#collision_model/HiddenBlock0/MatMulã
3collision_model/HiddenBlock0/BiasAdd/ReadVariableOpReadVariableOp<collision_model_hiddenblock0_biasadd_readvariableop_resource*
_output_shapes
:D*
dtype025
3collision_model/HiddenBlock0/BiasAdd/ReadVariableOpõ
$collision_model/HiddenBlock0/BiasAddBiasAdd-collision_model/HiddenBlock0/MatMul:product:0;collision_model/HiddenBlock0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD2&
$collision_model/HiddenBlock0/BiasAdd¸
$collision_model/HiddenBlock0/SigmoidSigmoid-collision_model/HiddenBlock0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD2&
$collision_model/HiddenBlock0/Sigmoidä
2collision_model/HiddenBlock1/MatMul/ReadVariableOpReadVariableOp;collision_model_hiddenblock1_matmul_readvariableop_resource*
_output_shapes

:DB*
dtype024
2collision_model/HiddenBlock1/MatMul/ReadVariableOpì
#collision_model/HiddenBlock1/MatMulMatMul(collision_model/HiddenBlock0/Sigmoid:y:0:collision_model/HiddenBlock1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2%
#collision_model/HiddenBlock1/MatMulã
3collision_model/HiddenBlock1/BiasAdd/ReadVariableOpReadVariableOp<collision_model_hiddenblock1_biasadd_readvariableop_resource*
_output_shapes
:B*
dtype025
3collision_model/HiddenBlock1/BiasAdd/ReadVariableOpõ
$collision_model/HiddenBlock1/BiasAddBiasAdd-collision_model/HiddenBlock1/MatMul:product:0;collision_model/HiddenBlock1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2&
$collision_model/HiddenBlock1/BiasAdd¸
$collision_model/HiddenBlock1/SigmoidSigmoid-collision_model/HiddenBlock1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2&
$collision_model/HiddenBlock1/Sigmoidä
2collision_model/HiddenBlock2/MatMul/ReadVariableOpReadVariableOp;collision_model_hiddenblock2_matmul_readvariableop_resource*
_output_shapes

:BA*
dtype024
2collision_model/HiddenBlock2/MatMul/ReadVariableOpì
#collision_model/HiddenBlock2/MatMulMatMul(collision_model/HiddenBlock1/Sigmoid:y:0:collision_model/HiddenBlock2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA2%
#collision_model/HiddenBlock2/MatMulã
3collision_model/HiddenBlock2/BiasAdd/ReadVariableOpReadVariableOp<collision_model_hiddenblock2_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype025
3collision_model/HiddenBlock2/BiasAdd/ReadVariableOpõ
$collision_model/HiddenBlock2/BiasAddBiasAdd-collision_model/HiddenBlock2/MatMul:product:0;collision_model/HiddenBlock2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA2&
$collision_model/HiddenBlock2/BiasAdd¸
$collision_model/HiddenBlock2/SigmoidSigmoid-collision_model/HiddenBlock2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA2&
$collision_model/HiddenBlock2/Sigmoidä
2collision_model/HiddenBlock3/MatMul/ReadVariableOpReadVariableOp;collision_model_hiddenblock3_matmul_readvariableop_resource*
_output_shapes

:A@*
dtype024
2collision_model/HiddenBlock3/MatMul/ReadVariableOpì
#collision_model/HiddenBlock3/MatMulMatMul(collision_model/HiddenBlock2/Sigmoid:y:0:collision_model/HiddenBlock3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#collision_model/HiddenBlock3/MatMulã
3collision_model/HiddenBlock3/BiasAdd/ReadVariableOpReadVariableOp<collision_model_hiddenblock3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3collision_model/HiddenBlock3/BiasAdd/ReadVariableOpõ
$collision_model/HiddenBlock3/BiasAddBiasAdd-collision_model/HiddenBlock3/MatMul:product:0;collision_model/HiddenBlock3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$collision_model/HiddenBlock3/BiasAdd¸
$collision_model/HiddenBlock3/SigmoidSigmoid-collision_model/HiddenBlock3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$collision_model/HiddenBlock3/SigmoidÆ
(collision_model/dG/MatMul/ReadVariableOpReadVariableOp1collision_model_dg_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02*
(collision_model/dG/MatMul/ReadVariableOpÎ
collision_model/dG/MatMulMatMul(collision_model/HiddenBlock3/Sigmoid:y:00collision_model/dG/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
collision_model/dG/MatMulÅ
)collision_model/dG/BiasAdd/ReadVariableOpReadVariableOp2collision_model_dg_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)collision_model/dG/BiasAdd/ReadVariableOpÍ
collision_model/dG/BiasAddBiasAdd#collision_model/dG/MatMul:product:01collision_model/dG/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
collision_model/dG/BiasAddÉ
)collision_model/tau/MatMul/ReadVariableOpReadVariableOp2collision_model_tau_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02+
)collision_model/tau/MatMul/ReadVariableOpÑ
collision_model/tau/MatMulMatMul(collision_model/HiddenBlock3/Sigmoid:y:01collision_model/tau/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
collision_model/tau/MatMulÈ
*collision_model/tau/BiasAdd/ReadVariableOpReadVariableOp3collision_model_tau_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*collision_model/tau/BiasAdd/ReadVariableOpÑ
collision_model/tau/BiasAddBiasAdd$collision_model/tau/MatMul:product:02collision_model/tau/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
collision_model/tau/BiasAddÓ
IdentityIdentity#collision_model/dG/BiasAdd:output:04^collision_model/HiddenBlock0/BiasAdd/ReadVariableOp3^collision_model/HiddenBlock0/MatMul/ReadVariableOp4^collision_model/HiddenBlock1/BiasAdd/ReadVariableOp3^collision_model/HiddenBlock1/MatMul/ReadVariableOp4^collision_model/HiddenBlock2/BiasAdd/ReadVariableOp3^collision_model/HiddenBlock2/MatMul/ReadVariableOp4^collision_model/HiddenBlock3/BiasAdd/ReadVariableOp3^collision_model/HiddenBlock3/MatMul/ReadVariableOp*^collision_model/dG/BiasAdd/ReadVariableOp)^collision_model/dG/MatMul/ReadVariableOp+^collision_model/tau/BiasAdd/ReadVariableOp*^collision_model/tau/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityØ

Identity_1Identity$collision_model/tau/BiasAdd:output:04^collision_model/HiddenBlock0/BiasAdd/ReadVariableOp3^collision_model/HiddenBlock0/MatMul/ReadVariableOp4^collision_model/HiddenBlock1/BiasAdd/ReadVariableOp3^collision_model/HiddenBlock1/MatMul/ReadVariableOp4^collision_model/HiddenBlock2/BiasAdd/ReadVariableOp3^collision_model/HiddenBlock2/MatMul/ReadVariableOp4^collision_model/HiddenBlock3/BiasAdd/ReadVariableOp3^collision_model/HiddenBlock3/MatMul/ReadVariableOp*^collision_model/dG/BiasAdd/ReadVariableOp)^collision_model/dG/MatMul/ReadVariableOp+^collision_model/tau/BiasAdd/ReadVariableOp*^collision_model/tau/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::2j
3collision_model/HiddenBlock0/BiasAdd/ReadVariableOp3collision_model/HiddenBlock0/BiasAdd/ReadVariableOp2h
2collision_model/HiddenBlock0/MatMul/ReadVariableOp2collision_model/HiddenBlock0/MatMul/ReadVariableOp2j
3collision_model/HiddenBlock1/BiasAdd/ReadVariableOp3collision_model/HiddenBlock1/BiasAdd/ReadVariableOp2h
2collision_model/HiddenBlock1/MatMul/ReadVariableOp2collision_model/HiddenBlock1/MatMul/ReadVariableOp2j
3collision_model/HiddenBlock2/BiasAdd/ReadVariableOp3collision_model/HiddenBlock2/BiasAdd/ReadVariableOp2h
2collision_model/HiddenBlock2/MatMul/ReadVariableOp2collision_model/HiddenBlock2/MatMul/ReadVariableOp2j
3collision_model/HiddenBlock3/BiasAdd/ReadVariableOp3collision_model/HiddenBlock3/BiasAdd/ReadVariableOp2h
2collision_model/HiddenBlock3/MatMul/ReadVariableOp2collision_model/HiddenBlock3/MatMul/ReadVariableOp2V
)collision_model/dG/BiasAdd/ReadVariableOp)collision_model/dG/BiasAdd/ReadVariableOp2T
(collision_model/dG/MatMul/ReadVariableOp(collision_model/dG/MatMul/ReadVariableOp2X
*collision_model/tau/BiasAdd/ReadVariableOp*collision_model/tau/BiasAdd/ReadVariableOp2V
)collision_model/tau/MatMul/ReadVariableOp)collision_model/tau/MatMul/ReadVariableOp:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namen:JF
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameT:JF
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameF
	
×
>__inference_dG_layer_call_and_return_conditional_losses_127159

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ô	
á
H__inference_HiddenBlock2_layer_call_and_return_conditional_losses_126550

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:BA*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿB::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB
 
_user_specified_nameinputs
Ð
x
#__inference_dG_layer_call_fn_127168

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_dG_layer_call_and_return_conditional_losses_1266032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Ø
?__inference_tau_layer_call_and_return_conditional_losses_127140

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
º

F__inference_InputBlock_layer_call_and_return_conditional_losses_127043
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2
ô	
á
H__inference_HiddenBlock1_layer_call_and_return_conditional_losses_127081

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:DB*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:B*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿD::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
 
_user_specified_nameinputs
èa
ê
__inference__traced_save_127341
file_prefix2
.savev2_hiddenblock0_kernel_read_readvariableop0
,savev2_hiddenblock0_bias_read_readvariableop2
.savev2_hiddenblock1_kernel_read_readvariableop0
,savev2_hiddenblock1_bias_read_readvariableop2
.savev2_hiddenblock2_kernel_read_readvariableop0
,savev2_hiddenblock2_bias_read_readvariableop2
.savev2_hiddenblock3_kernel_read_readvariableop0
,savev2_hiddenblock3_bias_read_readvariableop)
%savev2_tau_kernel_read_readvariableop'
#savev2_tau_bias_read_readvariableop(
$savev2_dg_kernel_read_readvariableop&
"savev2_dg_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop9
5savev2_adam_hiddenblock0_kernel_m_read_readvariableop7
3savev2_adam_hiddenblock0_bias_m_read_readvariableop9
5savev2_adam_hiddenblock1_kernel_m_read_readvariableop7
3savev2_adam_hiddenblock1_bias_m_read_readvariableop9
5savev2_adam_hiddenblock2_kernel_m_read_readvariableop7
3savev2_adam_hiddenblock2_bias_m_read_readvariableop9
5savev2_adam_hiddenblock3_kernel_m_read_readvariableop7
3savev2_adam_hiddenblock3_bias_m_read_readvariableop0
,savev2_adam_tau_kernel_m_read_readvariableop.
*savev2_adam_tau_bias_m_read_readvariableop/
+savev2_adam_dg_kernel_m_read_readvariableop-
)savev2_adam_dg_bias_m_read_readvariableop9
5savev2_adam_hiddenblock0_kernel_v_read_readvariableop7
3savev2_adam_hiddenblock0_bias_v_read_readvariableop9
5savev2_adam_hiddenblock1_kernel_v_read_readvariableop7
3savev2_adam_hiddenblock1_bias_v_read_readvariableop9
5savev2_adam_hiddenblock2_kernel_v_read_readvariableop7
3savev2_adam_hiddenblock2_bias_v_read_readvariableop9
5savev2_adam_hiddenblock3_kernel_v_read_readvariableop7
3savev2_adam_hiddenblock3_bias_v_read_readvariableop0
,savev2_adam_tau_kernel_v_read_readvariableop.
*savev2_adam_tau_bias_v_read_readvariableop/
+savev2_adam_dg_kernel_v_read_readvariableop-
)savev2_adam_dg_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*¤
valueB2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesì
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices¥
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_hiddenblock0_kernel_read_readvariableop,savev2_hiddenblock0_bias_read_readvariableop.savev2_hiddenblock1_kernel_read_readvariableop,savev2_hiddenblock1_bias_read_readvariableop.savev2_hiddenblock2_kernel_read_readvariableop,savev2_hiddenblock2_bias_read_readvariableop.savev2_hiddenblock3_kernel_read_readvariableop,savev2_hiddenblock3_bias_read_readvariableop%savev2_tau_kernel_read_readvariableop#savev2_tau_bias_read_readvariableop$savev2_dg_kernel_read_readvariableop"savev2_dg_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop5savev2_adam_hiddenblock0_kernel_m_read_readvariableop3savev2_adam_hiddenblock0_bias_m_read_readvariableop5savev2_adam_hiddenblock1_kernel_m_read_readvariableop3savev2_adam_hiddenblock1_bias_m_read_readvariableop5savev2_adam_hiddenblock2_kernel_m_read_readvariableop3savev2_adam_hiddenblock2_bias_m_read_readvariableop5savev2_adam_hiddenblock3_kernel_m_read_readvariableop3savev2_adam_hiddenblock3_bias_m_read_readvariableop,savev2_adam_tau_kernel_m_read_readvariableop*savev2_adam_tau_bias_m_read_readvariableop+savev2_adam_dg_kernel_m_read_readvariableop)savev2_adam_dg_bias_m_read_readvariableop5savev2_adam_hiddenblock0_kernel_v_read_readvariableop3savev2_adam_hiddenblock0_bias_v_read_readvariableop5savev2_adam_hiddenblock1_kernel_v_read_readvariableop3savev2_adam_hiddenblock1_bias_v_read_readvariableop5savev2_adam_hiddenblock2_kernel_v_read_readvariableop3savev2_adam_hiddenblock2_bias_v_read_readvariableop5savev2_adam_hiddenblock3_kernel_v_read_readvariableop3savev2_adam_hiddenblock3_bias_v_read_readvariableop,savev2_adam_tau_kernel_v_read_readvariableop*savev2_adam_tau_bias_v_read_readvariableop+savev2_adam_dg_kernel_v_read_readvariableop)savev2_adam_dg_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ó
_input_shapesÁ
¾: :D:D:DB:B:BA:A:A@:@:@::@:: : : : : : : : : : : : : :D:D:DB:B:BA:A:A@:@:@::@::D:D:DB:B:BA:A:A@:@:@::@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:D: 

_output_shapes
:D:$ 

_output_shapes

:DB: 

_output_shapes
:B:$ 

_output_shapes

:BA: 

_output_shapes
:A:$ 

_output_shapes

:A@: 

_output_shapes
:@:$	 

_output_shapes

:@: 


_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:D: 

_output_shapes
:D:$ 

_output_shapes

:DB: 

_output_shapes
:B:$ 

_output_shapes

:BA: 

_output_shapes
:A:$  

_output_shapes

:A@: !

_output_shapes
:@:$" 

_output_shapes

:@: #

_output_shapes
::$$ 

_output_shapes

:@: %

_output_shapes
::$& 

_output_shapes

:D: '

_output_shapes
:D:$( 

_output_shapes

:DB: )

_output_shapes
:B:$* 

_output_shapes

:BA: +

_output_shapes
:A:$, 

_output_shapes

:A@: -

_output_shapes
:@:$. 

_output_shapes

:@: /

_output_shapes
::$0 

_output_shapes

:@: 1

_output_shapes
::2

_output_shapes
: 
·A
à
K__inference_collision_model_layer_call_and_return_conditional_losses_126920
inputs_0
inputs_1
inputs_2/
+hiddenblock0_matmul_readvariableop_resource0
,hiddenblock0_biasadd_readvariableop_resource/
+hiddenblock1_matmul_readvariableop_resource0
,hiddenblock1_biasadd_readvariableop_resource/
+hiddenblock2_matmul_readvariableop_resource0
,hiddenblock2_biasadd_readvariableop_resource/
+hiddenblock3_matmul_readvariableop_resource0
,hiddenblock3_biasadd_readvariableop_resource%
!dg_matmul_readvariableop_resource&
"dg_biasadd_readvariableop_resource&
"tau_matmul_readvariableop_resource'
#tau_biasadd_readvariableop_resource
identity

identity_1¢#HiddenBlock0/BiasAdd/ReadVariableOp¢"HiddenBlock0/MatMul/ReadVariableOp¢#HiddenBlock1/BiasAdd/ReadVariableOp¢"HiddenBlock1/MatMul/ReadVariableOp¢#HiddenBlock2/BiasAdd/ReadVariableOp¢"HiddenBlock2/MatMul/ReadVariableOp¢#HiddenBlock3/BiasAdd/ReadVariableOp¢"HiddenBlock3/MatMul/ReadVariableOp¢dG/BiasAdd/ReadVariableOp¢dG/MatMul/ReadVariableOp¢tau/BiasAdd/ReadVariableOp¢tau/MatMul/ReadVariableOpr
InputBlock/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
InputBlock/concat/axis¬
InputBlock/concatConcatV2inputs_0inputs_1inputs_2InputBlock/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
InputBlock/concat´
"HiddenBlock0/MatMul/ReadVariableOpReadVariableOp+hiddenblock0_matmul_readvariableop_resource*
_output_shapes

:D*
dtype02$
"HiddenBlock0/MatMul/ReadVariableOp®
HiddenBlock0/MatMulMatMulInputBlock/concat:output:0*HiddenBlock0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD2
HiddenBlock0/MatMul³
#HiddenBlock0/BiasAdd/ReadVariableOpReadVariableOp,hiddenblock0_biasadd_readvariableop_resource*
_output_shapes
:D*
dtype02%
#HiddenBlock0/BiasAdd/ReadVariableOpµ
HiddenBlock0/BiasAddBiasAddHiddenBlock0/MatMul:product:0+HiddenBlock0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD2
HiddenBlock0/BiasAdd
HiddenBlock0/SigmoidSigmoidHiddenBlock0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD2
HiddenBlock0/Sigmoid´
"HiddenBlock1/MatMul/ReadVariableOpReadVariableOp+hiddenblock1_matmul_readvariableop_resource*
_output_shapes

:DB*
dtype02$
"HiddenBlock1/MatMul/ReadVariableOp¬
HiddenBlock1/MatMulMatMulHiddenBlock0/Sigmoid:y:0*HiddenBlock1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2
HiddenBlock1/MatMul³
#HiddenBlock1/BiasAdd/ReadVariableOpReadVariableOp,hiddenblock1_biasadd_readvariableop_resource*
_output_shapes
:B*
dtype02%
#HiddenBlock1/BiasAdd/ReadVariableOpµ
HiddenBlock1/BiasAddBiasAddHiddenBlock1/MatMul:product:0+HiddenBlock1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2
HiddenBlock1/BiasAdd
HiddenBlock1/SigmoidSigmoidHiddenBlock1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2
HiddenBlock1/Sigmoid´
"HiddenBlock2/MatMul/ReadVariableOpReadVariableOp+hiddenblock2_matmul_readvariableop_resource*
_output_shapes

:BA*
dtype02$
"HiddenBlock2/MatMul/ReadVariableOp¬
HiddenBlock2/MatMulMatMulHiddenBlock1/Sigmoid:y:0*HiddenBlock2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA2
HiddenBlock2/MatMul³
#HiddenBlock2/BiasAdd/ReadVariableOpReadVariableOp,hiddenblock2_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02%
#HiddenBlock2/BiasAdd/ReadVariableOpµ
HiddenBlock2/BiasAddBiasAddHiddenBlock2/MatMul:product:0+HiddenBlock2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA2
HiddenBlock2/BiasAdd
HiddenBlock2/SigmoidSigmoidHiddenBlock2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA2
HiddenBlock2/Sigmoid´
"HiddenBlock3/MatMul/ReadVariableOpReadVariableOp+hiddenblock3_matmul_readvariableop_resource*
_output_shapes

:A@*
dtype02$
"HiddenBlock3/MatMul/ReadVariableOp¬
HiddenBlock3/MatMulMatMulHiddenBlock2/Sigmoid:y:0*HiddenBlock3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
HiddenBlock3/MatMul³
#HiddenBlock3/BiasAdd/ReadVariableOpReadVariableOp,hiddenblock3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#HiddenBlock3/BiasAdd/ReadVariableOpµ
HiddenBlock3/BiasAddBiasAddHiddenBlock3/MatMul:product:0+HiddenBlock3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
HiddenBlock3/BiasAdd
HiddenBlock3/SigmoidSigmoidHiddenBlock3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
HiddenBlock3/Sigmoid
dG/MatMul/ReadVariableOpReadVariableOp!dg_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dG/MatMul/ReadVariableOp
	dG/MatMulMatMulHiddenBlock3/Sigmoid:y:0 dG/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dG/MatMul
dG/BiasAdd/ReadVariableOpReadVariableOp"dg_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dG/BiasAdd/ReadVariableOp

dG/BiasAddBiasAdddG/MatMul:product:0!dG/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dG/BiasAdd
tau/MatMul/ReadVariableOpReadVariableOp"tau_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
tau/MatMul/ReadVariableOp

tau/MatMulMatMulHiddenBlock3/Sigmoid:y:0!tau/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

tau/MatMul
tau/BiasAdd/ReadVariableOpReadVariableOp#tau_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
tau/BiasAdd/ReadVariableOp
tau/BiasAddBiasAddtau/MatMul:product:0"tau/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tau/BiasAdd
IdentityIdentitytau/BiasAdd:output:0$^HiddenBlock0/BiasAdd/ReadVariableOp#^HiddenBlock0/MatMul/ReadVariableOp$^HiddenBlock1/BiasAdd/ReadVariableOp#^HiddenBlock1/MatMul/ReadVariableOp$^HiddenBlock2/BiasAdd/ReadVariableOp#^HiddenBlock2/MatMul/ReadVariableOp$^HiddenBlock3/BiasAdd/ReadVariableOp#^HiddenBlock3/MatMul/ReadVariableOp^dG/BiasAdd/ReadVariableOp^dG/MatMul/ReadVariableOp^tau/BiasAdd/ReadVariableOp^tau/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1IdentitydG/BiasAdd:output:0$^HiddenBlock0/BiasAdd/ReadVariableOp#^HiddenBlock0/MatMul/ReadVariableOp$^HiddenBlock1/BiasAdd/ReadVariableOp#^HiddenBlock1/MatMul/ReadVariableOp$^HiddenBlock2/BiasAdd/ReadVariableOp#^HiddenBlock2/MatMul/ReadVariableOp$^HiddenBlock3/BiasAdd/ReadVariableOp#^HiddenBlock3/MatMul/ReadVariableOp^dG/BiasAdd/ReadVariableOp^dG/MatMul/ReadVariableOp^tau/BiasAdd/ReadVariableOp^tau/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::2J
#HiddenBlock0/BiasAdd/ReadVariableOp#HiddenBlock0/BiasAdd/ReadVariableOp2H
"HiddenBlock0/MatMul/ReadVariableOp"HiddenBlock0/MatMul/ReadVariableOp2J
#HiddenBlock1/BiasAdd/ReadVariableOp#HiddenBlock1/BiasAdd/ReadVariableOp2H
"HiddenBlock1/MatMul/ReadVariableOp"HiddenBlock1/MatMul/ReadVariableOp2J
#HiddenBlock2/BiasAdd/ReadVariableOp#HiddenBlock2/BiasAdd/ReadVariableOp2H
"HiddenBlock2/MatMul/ReadVariableOp"HiddenBlock2/MatMul/ReadVariableOp2J
#HiddenBlock3/BiasAdd/ReadVariableOp#HiddenBlock3/BiasAdd/ReadVariableOp2H
"HiddenBlock3/MatMul/ReadVariableOp"HiddenBlock3/MatMul/ReadVariableOp26
dG/BiasAdd/ReadVariableOpdG/BiasAdd/ReadVariableOp24
dG/MatMul/ReadVariableOpdG/MatMul/ReadVariableOp28
tau/BiasAdd/ReadVariableOptau/BiasAdd/ReadVariableOp26
tau/MatMul/ReadVariableOptau/MatMul/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2
å

-__inference_HiddenBlock1_layer_call_fn_127090

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock1_layer_call_and_return_conditional_losses_1265232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿD::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
 
_user_specified_nameinputs

Ê
0__inference_collision_model_layer_call_fn_127002
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_collision_model_layer_call_and_return_conditional_losses_1267282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2
å

-__inference_HiddenBlock3_layer_call_fn_127130

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_HiddenBlock3_layer_call_and_return_conditional_losses_1265772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿA::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
ô	
á
H__inference_HiddenBlock0_layer_call_and_return_conditional_losses_126496

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:D*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:D*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·A
à
K__inference_collision_model_layer_call_and_return_conditional_losses_126969
inputs_0
inputs_1
inputs_2/
+hiddenblock0_matmul_readvariableop_resource0
,hiddenblock0_biasadd_readvariableop_resource/
+hiddenblock1_matmul_readvariableop_resource0
,hiddenblock1_biasadd_readvariableop_resource/
+hiddenblock2_matmul_readvariableop_resource0
,hiddenblock2_biasadd_readvariableop_resource/
+hiddenblock3_matmul_readvariableop_resource0
,hiddenblock3_biasadd_readvariableop_resource%
!dg_matmul_readvariableop_resource&
"dg_biasadd_readvariableop_resource&
"tau_matmul_readvariableop_resource'
#tau_biasadd_readvariableop_resource
identity

identity_1¢#HiddenBlock0/BiasAdd/ReadVariableOp¢"HiddenBlock0/MatMul/ReadVariableOp¢#HiddenBlock1/BiasAdd/ReadVariableOp¢"HiddenBlock1/MatMul/ReadVariableOp¢#HiddenBlock2/BiasAdd/ReadVariableOp¢"HiddenBlock2/MatMul/ReadVariableOp¢#HiddenBlock3/BiasAdd/ReadVariableOp¢"HiddenBlock3/MatMul/ReadVariableOp¢dG/BiasAdd/ReadVariableOp¢dG/MatMul/ReadVariableOp¢tau/BiasAdd/ReadVariableOp¢tau/MatMul/ReadVariableOpr
InputBlock/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
InputBlock/concat/axis¬
InputBlock/concatConcatV2inputs_0inputs_1inputs_2InputBlock/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
InputBlock/concat´
"HiddenBlock0/MatMul/ReadVariableOpReadVariableOp+hiddenblock0_matmul_readvariableop_resource*
_output_shapes

:D*
dtype02$
"HiddenBlock0/MatMul/ReadVariableOp®
HiddenBlock0/MatMulMatMulInputBlock/concat:output:0*HiddenBlock0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD2
HiddenBlock0/MatMul³
#HiddenBlock0/BiasAdd/ReadVariableOpReadVariableOp,hiddenblock0_biasadd_readvariableop_resource*
_output_shapes
:D*
dtype02%
#HiddenBlock0/BiasAdd/ReadVariableOpµ
HiddenBlock0/BiasAddBiasAddHiddenBlock0/MatMul:product:0+HiddenBlock0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD2
HiddenBlock0/BiasAdd
HiddenBlock0/SigmoidSigmoidHiddenBlock0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD2
HiddenBlock0/Sigmoid´
"HiddenBlock1/MatMul/ReadVariableOpReadVariableOp+hiddenblock1_matmul_readvariableop_resource*
_output_shapes

:DB*
dtype02$
"HiddenBlock1/MatMul/ReadVariableOp¬
HiddenBlock1/MatMulMatMulHiddenBlock0/Sigmoid:y:0*HiddenBlock1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2
HiddenBlock1/MatMul³
#HiddenBlock1/BiasAdd/ReadVariableOpReadVariableOp,hiddenblock1_biasadd_readvariableop_resource*
_output_shapes
:B*
dtype02%
#HiddenBlock1/BiasAdd/ReadVariableOpµ
HiddenBlock1/BiasAddBiasAddHiddenBlock1/MatMul:product:0+HiddenBlock1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2
HiddenBlock1/BiasAdd
HiddenBlock1/SigmoidSigmoidHiddenBlock1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB2
HiddenBlock1/Sigmoid´
"HiddenBlock2/MatMul/ReadVariableOpReadVariableOp+hiddenblock2_matmul_readvariableop_resource*
_output_shapes

:BA*
dtype02$
"HiddenBlock2/MatMul/ReadVariableOp¬
HiddenBlock2/MatMulMatMulHiddenBlock1/Sigmoid:y:0*HiddenBlock2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA2
HiddenBlock2/MatMul³
#HiddenBlock2/BiasAdd/ReadVariableOpReadVariableOp,hiddenblock2_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype02%
#HiddenBlock2/BiasAdd/ReadVariableOpµ
HiddenBlock2/BiasAddBiasAddHiddenBlock2/MatMul:product:0+HiddenBlock2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA2
HiddenBlock2/BiasAdd
HiddenBlock2/SigmoidSigmoidHiddenBlock2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA2
HiddenBlock2/Sigmoid´
"HiddenBlock3/MatMul/ReadVariableOpReadVariableOp+hiddenblock3_matmul_readvariableop_resource*
_output_shapes

:A@*
dtype02$
"HiddenBlock3/MatMul/ReadVariableOp¬
HiddenBlock3/MatMulMatMulHiddenBlock2/Sigmoid:y:0*HiddenBlock3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
HiddenBlock3/MatMul³
#HiddenBlock3/BiasAdd/ReadVariableOpReadVariableOp,hiddenblock3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#HiddenBlock3/BiasAdd/ReadVariableOpµ
HiddenBlock3/BiasAddBiasAddHiddenBlock3/MatMul:product:0+HiddenBlock3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
HiddenBlock3/BiasAdd
HiddenBlock3/SigmoidSigmoidHiddenBlock3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
HiddenBlock3/Sigmoid
dG/MatMul/ReadVariableOpReadVariableOp!dg_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dG/MatMul/ReadVariableOp
	dG/MatMulMatMulHiddenBlock3/Sigmoid:y:0 dG/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	dG/MatMul
dG/BiasAdd/ReadVariableOpReadVariableOp"dg_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dG/BiasAdd/ReadVariableOp

dG/BiasAddBiasAdddG/MatMul:product:0!dG/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dG/BiasAdd
tau/MatMul/ReadVariableOpReadVariableOp"tau_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
tau/MatMul/ReadVariableOp

tau/MatMulMatMulHiddenBlock3/Sigmoid:y:0!tau/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

tau/MatMul
tau/BiasAdd/ReadVariableOpReadVariableOp#tau_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
tau/BiasAdd/ReadVariableOp
tau/BiasAddBiasAddtau/MatMul:product:0"tau/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tau/BiasAdd
IdentityIdentitytau/BiasAdd:output:0$^HiddenBlock0/BiasAdd/ReadVariableOp#^HiddenBlock0/MatMul/ReadVariableOp$^HiddenBlock1/BiasAdd/ReadVariableOp#^HiddenBlock1/MatMul/ReadVariableOp$^HiddenBlock2/BiasAdd/ReadVariableOp#^HiddenBlock2/MatMul/ReadVariableOp$^HiddenBlock3/BiasAdd/ReadVariableOp#^HiddenBlock3/MatMul/ReadVariableOp^dG/BiasAdd/ReadVariableOp^dG/MatMul/ReadVariableOp^tau/BiasAdd/ReadVariableOp^tau/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1IdentitydG/BiasAdd:output:0$^HiddenBlock0/BiasAdd/ReadVariableOp#^HiddenBlock0/MatMul/ReadVariableOp$^HiddenBlock1/BiasAdd/ReadVariableOp#^HiddenBlock1/MatMul/ReadVariableOp$^HiddenBlock2/BiasAdd/ReadVariableOp#^HiddenBlock2/MatMul/ReadVariableOp$^HiddenBlock3/BiasAdd/ReadVariableOp#^HiddenBlock3/MatMul/ReadVariableOp^dG/BiasAdd/ReadVariableOp^dG/MatMul/ReadVariableOp^tau/BiasAdd/ReadVariableOp^tau/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::2J
#HiddenBlock0/BiasAdd/ReadVariableOp#HiddenBlock0/BiasAdd/ReadVariableOp2H
"HiddenBlock0/MatMul/ReadVariableOp"HiddenBlock0/MatMul/ReadVariableOp2J
#HiddenBlock1/BiasAdd/ReadVariableOp#HiddenBlock1/BiasAdd/ReadVariableOp2H
"HiddenBlock1/MatMul/ReadVariableOp"HiddenBlock1/MatMul/ReadVariableOp2J
#HiddenBlock2/BiasAdd/ReadVariableOp#HiddenBlock2/BiasAdd/ReadVariableOp2H
"HiddenBlock2/MatMul/ReadVariableOp"HiddenBlock2/MatMul/ReadVariableOp2J
#HiddenBlock3/BiasAdd/ReadVariableOp#HiddenBlock3/BiasAdd/ReadVariableOp2H
"HiddenBlock3/MatMul/ReadVariableOp"HiddenBlock3/MatMul/ReadVariableOp26
dG/BiasAdd/ReadVariableOpdG/BiasAdd/ReadVariableOp24
dG/MatMul/ReadVariableOpdG/MatMul/ReadVariableOp28
tau/BiasAdd/ReadVariableOptau/BiasAdd/ReadVariableOp26
tau/MatMul/ReadVariableOptau/MatMul/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2
ô	
á
H__inference_HiddenBlock2_layer_call_and_return_conditional_losses_127101

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:BA*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:A*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿB::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*´
serving_default 
/
F*
serving_default_F:0ÿÿÿÿÿÿÿÿÿ
/
T*
serving_default_T:0ÿÿÿÿÿÿÿÿÿ
/
n*
serving_default_n:0ÿÿÿÿÿÿÿÿÿ6
dG0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ7
tau0
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:
ÃO
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
	optimizer
loss
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+&call_and_return_all_conditional_losses
_default_save_signature
__call__"ÍK
_tf_keras_network±K{"class_name": "Functional", "name": "collision_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "collision_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "n"}, "name": "n", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "T"}, "name": "T", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 15]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "F"}, "name": "F", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "InputBlock", "trainable": true, "dtype": "float32", "axis": -1}, "name": "InputBlock", "inbound_nodes": [[["n", 0, 0, {}], ["T", 0, 0, {}], ["F", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "HiddenBlock0", "trainable": true, "dtype": "float32", "units": 68, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "HiddenBlock0", "inbound_nodes": [[["InputBlock", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "HiddenBlock1", "trainable": true, "dtype": "float32", "units": 66, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "HiddenBlock1", "inbound_nodes": [[["HiddenBlock0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "HiddenBlock2", "trainable": true, "dtype": "float32", "units": 65, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "HiddenBlock2", "inbound_nodes": [[["HiddenBlock1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "HiddenBlock3", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "HiddenBlock3", "inbound_nodes": [[["HiddenBlock2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "tau", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "tau", "inbound_nodes": [[["HiddenBlock3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dG", "trainable": true, "dtype": "float32", "units": 15, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dG", "inbound_nodes": [[["HiddenBlock3", 0, 0, {}]]]}], "input_layers": [["n", 0, 0], ["T", 0, 0], ["F", 0, 0]], "output_layers": [["tau", 0, 0], ["dG", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 15]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 15]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "collision_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "n"}, "name": "n", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "T"}, "name": "T", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 15]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "F"}, "name": "F", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "InputBlock", "trainable": true, "dtype": "float32", "axis": -1}, "name": "InputBlock", "inbound_nodes": [[["n", 0, 0, {}], ["T", 0, 0, {}], ["F", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "HiddenBlock0", "trainable": true, "dtype": "float32", "units": 68, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "HiddenBlock0", "inbound_nodes": [[["InputBlock", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "HiddenBlock1", "trainable": true, "dtype": "float32", "units": 66, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "HiddenBlock1", "inbound_nodes": [[["HiddenBlock0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "HiddenBlock2", "trainable": true, "dtype": "float32", "units": 65, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "HiddenBlock2", "inbound_nodes": [[["HiddenBlock1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "HiddenBlock3", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "HiddenBlock3", "inbound_nodes": [[["HiddenBlock2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "tau", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "tau", "inbound_nodes": [[["HiddenBlock3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dG", "trainable": true, "dtype": "float32", "units": 15, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dG", "inbound_nodes": [[["HiddenBlock3", 0, 0, {}]]]}], "input_layers": [["n", 0, 0], ["T", 0, 0], ["F", 0, 0]], "output_layers": [["tau", 0, 0], ["dG", 0, 0]]}}, "training_config": {"loss": {"tau": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "dG": "dg_loss"}, "metrics": [[null], [{"class_name": "MeanMetricWrapper", "config": {"name": "dG_mean_squared_error", "dtype": "float32", "fn": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}, "__passive_serialization__": true}}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Ý"Ú
_tf_keras_input_layerº{"class_name": "InputLayer", "name": "n", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "n"}}
Ý"Ú
_tf_keras_input_layerº{"class_name": "InputLayer", "name": "T", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "T"}}
ß"Ü
_tf_keras_input_layer¼{"class_name": "InputLayer", "name": "F", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 15]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "F"}}
û
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"ê
_tf_keras_layerÐ{"class_name": "Concatenate", "name": "InputBlock", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "InputBlock", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 15]}]}
û

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ô
_tf_keras_layerº{"class_name": "Dense", "name": "HiddenBlock0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "HiddenBlock0", "trainable": true, "dtype": "float32", "units": 68, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 17}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17]}}
û

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
+&call_and_return_all_conditional_losses
__call__"Ô
_tf_keras_layerº{"class_name": "Dense", "name": "HiddenBlock1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "HiddenBlock1", "trainable": true, "dtype": "float32", "units": 66, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 68}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 68]}}
û

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
+&call_and_return_all_conditional_losses
__call__"Ô
_tf_keras_layerº{"class_name": "Dense", "name": "HiddenBlock2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "HiddenBlock2", "trainable": true, "dtype": "float32", "units": 65, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 66}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66]}}
û

(kernel
)bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
+&call_and_return_all_conditional_losses
 __call__"Ô
_tf_keras_layerº{"class_name": "Dense", "name": "HiddenBlock3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "HiddenBlock3", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 65}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 65]}}
ç

.kernel
/bias
0trainable_variables
1regularization_losses
2	variables
3	keras_api
+¡&call_and_return_all_conditional_losses
¢__call__"À
_tf_keras_layer¦{"class_name": "Dense", "name": "tau", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "tau", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
æ

4kernel
5bias
6trainable_variables
7regularization_losses
8	variables
9	keras_api
+£&call_and_return_all_conditional_losses
¤__call__"¿
_tf_keras_layer¥{"class_name": "Dense", "name": "dG", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dG", "trainable": true, "dtype": "float32", "units": 15, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
¿
:iter

;beta_1

<beta_2
	=decay
>learning_ratem|m}m~m"m#m(m)m.m/m4m5mvvvv"v#v(v)v.v/v4v5v"
	optimizer
 "
trackable_dict_wrapper
v
0
1
2
3
"4
#5
(6
)7
.8
/9
410
511"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
"4
#5
(6
)7
.8
/9
410
511"
trackable_list_wrapper
Î
trainable_variables
regularization_losses
?metrics
@layer_metrics

Alayers
	variables
Bnon_trainable_variables
Clayer_regularization_losses
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
¥serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
trainable_variables
Dlayer_metrics
Emetrics
regularization_losses

Flayers
	variables
Gnon_trainable_variables
Hlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
%:#D2HiddenBlock0/kernel
:D2HiddenBlock0/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
trainable_variables
Ilayer_metrics
Jmetrics
regularization_losses

Klayers
	variables
Lnon_trainable_variables
Mlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
%:#DB2HiddenBlock1/kernel
:B2HiddenBlock1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
trainable_variables
Nlayer_metrics
Ometrics
regularization_losses

Players
 	variables
Qnon_trainable_variables
Rlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
%:#BA2HiddenBlock2/kernel
:A2HiddenBlock2/bias
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
°
$trainable_variables
Slayer_metrics
Tmetrics
%regularization_losses

Ulayers
&	variables
Vnon_trainable_variables
Wlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
%:#A@2HiddenBlock3/kernel
:@2HiddenBlock3/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
°
*trainable_variables
Xlayer_metrics
Ymetrics
+regularization_losses

Zlayers
,	variables
[non_trainable_variables
\layer_regularization_losses
 __call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:@2
tau/kernel
:2tau/bias
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
°
0trainable_variables
]layer_metrics
^metrics
1regularization_losses

_layers
2	variables
`non_trainable_variables
alayer_regularization_losses
¢__call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
:@2	dG/kernel
:2dG/bias
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
°
6trainable_variables
blayer_metrics
cmetrics
7regularization_losses

dlayers
8	variables
enon_trainable_variables
flayer_regularization_losses
¤__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
<
g0
h1
i2
j3"
trackable_list_wrapper
 "
trackable_dict_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
»
	ktotal
	lcount
m	variables
n	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Ã
	ototal
	pcount
q	variables
r	keras_api"
_tf_keras_metricr{"class_name": "Mean", "name": "tau_loss", "dtype": "float32", "config": {"name": "tau_loss", "dtype": "float32"}}
Á
	stotal
	tcount
u	variables
v	keras_api"
_tf_keras_metricp{"class_name": "Mean", "name": "dG_loss", "dtype": "float32", "config": {"name": "dG_loss", "dtype": "float32"}}

	wtotal
	xcount
y
_fn_kwargs
z	variables
{	keras_api"À
_tf_keras_metric¥{"class_name": "MeanMetricWrapper", "name": "dG_mean_squared_error", "dtype": "float32", "config": {"name": "dG_mean_squared_error", "dtype": "float32", "fn": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}, "__passive_serialization__": true}}}
:  (2total
:  (2count
.
k0
l1"
trackable_list_wrapper
-
m	variables"
_generic_user_object
:  (2total
:  (2count
.
o0
p1"
trackable_list_wrapper
-
q	variables"
_generic_user_object
:  (2total
:  (2count
.
s0
t1"
trackable_list_wrapper
-
u	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
w0
x1"
trackable_list_wrapper
-
z	variables"
_generic_user_object
*:(D2Adam/HiddenBlock0/kernel/m
$:"D2Adam/HiddenBlock0/bias/m
*:(DB2Adam/HiddenBlock1/kernel/m
$:"B2Adam/HiddenBlock1/bias/m
*:(BA2Adam/HiddenBlock2/kernel/m
$:"A2Adam/HiddenBlock2/bias/m
*:(A@2Adam/HiddenBlock3/kernel/m
$:"@2Adam/HiddenBlock3/bias/m
!:@2Adam/tau/kernel/m
:2Adam/tau/bias/m
 :@2Adam/dG/kernel/m
:2Adam/dG/bias/m
*:(D2Adam/HiddenBlock0/kernel/v
$:"D2Adam/HiddenBlock0/bias/v
*:(DB2Adam/HiddenBlock1/kernel/v
$:"B2Adam/HiddenBlock1/bias/v
*:(BA2Adam/HiddenBlock2/kernel/v
$:"A2Adam/HiddenBlock2/bias/v
*:(A@2Adam/HiddenBlock3/kernel/v
$:"@2Adam/HiddenBlock3/bias/v
!:@2Adam/tau/kernel/v
:2Adam/tau/bias/v
 :@2Adam/dG/kernel/v
:2Adam/dG/bias/v
ú2÷
K__inference_collision_model_layer_call_and_return_conditional_losses_126969
K__inference_collision_model_layer_call_and_return_conditional_losses_126685
K__inference_collision_model_layer_call_and_return_conditional_losses_126647
K__inference_collision_model_layer_call_and_return_conditional_losses_126920À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
!__inference__wrapped_model_126461ï
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *_¢\
ZW

nÿÿÿÿÿÿÿÿÿ

Tÿÿÿÿÿÿÿÿÿ

Fÿÿÿÿÿÿÿÿÿ
2
0__inference_collision_model_layer_call_fn_126828
0__inference_collision_model_layer_call_fn_127035
0__inference_collision_model_layer_call_fn_126757
0__inference_collision_model_layer_call_fn_127002À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ð2í
F__inference_InputBlock_layer_call_and_return_conditional_losses_127043¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_InputBlock_layer_call_fn_127050¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_HiddenBlock0_layer_call_and_return_conditional_losses_127061¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_HiddenBlock0_layer_call_fn_127070¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_HiddenBlock1_layer_call_and_return_conditional_losses_127081¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_HiddenBlock1_layer_call_fn_127090¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_HiddenBlock2_layer_call_and_return_conditional_losses_127101¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_HiddenBlock2_layer_call_fn_127110¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_HiddenBlock3_layer_call_and_return_conditional_losses_127121¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_HiddenBlock3_layer_call_fn_127130¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
?__inference_tau_layer_call_and_return_conditional_losses_127140¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Î2Ë
$__inference_tau_layer_call_fn_127149¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
è2å
>__inference_dG_layer_call_and_return_conditional_losses_127159¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Í2Ê
#__inference_dG_layer_call_fn_127168¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÉBÆ
$__inference_signature_wrapper_126871FTn"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ¨
H__inference_HiddenBlock0_layer_call_and_return_conditional_losses_127061\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿD
 
-__inference_HiddenBlock0_layer_call_fn_127070O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿD¨
H__inference_HiddenBlock1_layer_call_and_return_conditional_losses_127081\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿD
ª "%¢"

0ÿÿÿÿÿÿÿÿÿB
 
-__inference_HiddenBlock1_layer_call_fn_127090O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿD
ª "ÿÿÿÿÿÿÿÿÿB¨
H__inference_HiddenBlock2_layer_call_and_return_conditional_losses_127101\"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿB
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
-__inference_HiddenBlock2_layer_call_fn_127110O"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿB
ª "ÿÿÿÿÿÿÿÿÿA¨
H__inference_HiddenBlock3_layer_call_and_return_conditional_losses_127121\()/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
-__inference_HiddenBlock3_layer_call_fn_127130O()/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿ@ò
F__inference_InputBlock_layer_call_and_return_conditional_losses_127043§~¢{
t¢q
ol
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ê
+__inference_InputBlock_layer_call_fn_127050~¢{
t¢q
ol
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿî
!__inference__wrapped_model_126461È"#()45./i¢f
_¢\
ZW

nÿÿÿÿÿÿÿÿÿ

Tÿÿÿÿÿÿÿÿÿ

Fÿÿÿÿÿÿÿÿÿ
ª "MªJ
"
dG
dGÿÿÿÿÿÿÿÿÿ
$
tau
tauÿÿÿÿÿÿÿÿÿ
K__inference_collision_model_layer_call_and_return_conditional_losses_126647Î"#()45./q¢n
g¢d
ZW

nÿÿÿÿÿÿÿÿÿ

Tÿÿÿÿÿÿÿÿÿ

Fÿÿÿÿÿÿÿÿÿ
p

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 
K__inference_collision_model_layer_call_and_return_conditional_losses_126685Î"#()45./q¢n
g¢d
ZW

nÿÿÿÿÿÿÿÿÿ

Tÿÿÿÿÿÿÿÿÿ

Fÿÿÿÿÿÿÿÿÿ
p 

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 µ
K__inference_collision_model_layer_call_and_return_conditional_losses_126920å"#()45./¢
|¢y
ol
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
p

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 µ
K__inference_collision_model_layer_call_and_return_conditional_losses_126969å"#()45./¢
|¢y
ol
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 õ
0__inference_collision_model_layer_call_fn_126757À"#()45./q¢n
g¢d
ZW

nÿÿÿÿÿÿÿÿÿ

Tÿÿÿÿÿÿÿÿÿ

Fÿÿÿÿÿÿÿÿÿ
p

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿõ
0__inference_collision_model_layer_call_fn_126828À"#()45./q¢n
g¢d
ZW

nÿÿÿÿÿÿÿÿÿ

Tÿÿÿÿÿÿÿÿÿ

Fÿÿÿÿÿÿÿÿÿ
p 

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ
0__inference_collision_model_layer_call_fn_127002×"#()45./¢
|¢y
ol
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
p

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ
0__inference_collision_model_layer_call_fn_127035×"#()45./¢
|¢y
ol
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
"
inputs/2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ
>__inference_dG_layer_call_and_return_conditional_losses_127159\45/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 v
#__inference_dG_layer_call_fn_127168O45/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿû
$__inference_signature_wrapper_126871Ò"#()45./s¢p
¢ 
iªf
 
F
Fÿÿÿÿÿÿÿÿÿ
 
T
Tÿÿÿÿÿÿÿÿÿ
 
n
nÿÿÿÿÿÿÿÿÿ"MªJ
"
dG
dGÿÿÿÿÿÿÿÿÿ
$
tau
tauÿÿÿÿÿÿÿÿÿ
?__inference_tau_layer_call_and_return_conditional_losses_127140\.//¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 w
$__inference_tau_layer_call_fn_127149O.//¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ