??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

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
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ??
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
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
?
block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel
?
'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0
?
block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel
?
'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0
?
block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*$
shared_nameblock2_conv1/kernel
?
'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@?*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:?*
dtype0
?
block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock2_conv2/kernel
?
'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:?*
dtype0
?
block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv1/kernel
?
'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:?*
dtype0
?
block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv2/kernel
?
'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:?*
dtype0
?
block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv3/kernel
?
'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:?*
dtype0
?
block3_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv4/kernel
?
'block3_conv4/kernel/Read/ReadVariableOpReadVariableOpblock3_conv4/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv4/bias
t
%block3_conv4/bias/Read/ReadVariableOpReadVariableOpblock3_conv4/bias*
_output_shapes	
:?*
dtype0
?
block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv1/kernel
?
'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:?*
dtype0
?
block4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv2/kernel
?
'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:?*
dtype0
?
block4_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv3/kernel
?
'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:?*
dtype0
?
block4_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv4/kernel
?
'block4_conv4/kernel/Read/ReadVariableOpReadVariableOpblock4_conv4/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv4/bias
t
%block4_conv4/bias/Read/ReadVariableOpReadVariableOpblock4_conv4/bias*
_output_shapes	
:?*
dtype0
?
block5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv1/kernel
?
'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:?*
dtype0
?
block5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv2/kernel
?
'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
_output_shapes	
:?*
dtype0
?
block5_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv3/kernel
?
'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv3/bias
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
_output_shapes	
:?*
dtype0
?
block5_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv4/kernel
?
'block5_conv4/kernel/Read/ReadVariableOpReadVariableOpblock5_conv4/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv4/bias
t
%block5_conv4/bias/Read/ReadVariableOpReadVariableOpblock5_conv4/bias*
_output_shapes	
:?*
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
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	?@*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	?@*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?p
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?o
value?oB?o B?o
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
?

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer_with_weights-11
layer-15
layer-16
layer_with_weights-12
layer-17
layer_with_weights-13
layer-18
layer_with_weights-14
layer-19
layer_with_weights-15
layer-20
layer-21
 	variables
!trainable_variables
"regularization_losses
#	keras_api
R
$	variables
%trainable_variables
&regularization_losses
'	keras_api
h

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
h
.iter

/beta_1

0beta_2
	1decay
2learning_rate(m?)m?(v?)v?
?
30
41
52
63
74
85
96
:7
;8
<9
=10
>11
?12
@13
A14
B15
C16
D17
E18
F19
G20
H21
I22
J23
K24
L25
M26
N27
O28
P29
Q30
R31
(32
)33

(0
)1
 
?
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
h

3kernel
4bias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
h

5kernel
6bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
R
`	variables
atrainable_variables
bregularization_losses
c	keras_api
h

7kernel
8bias
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h

9kernel
:bias
h	variables
itrainable_variables
jregularization_losses
k	keras_api
R
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
h

;kernel
<bias
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
h

=kernel
>bias
t	variables
utrainable_variables
vregularization_losses
w	keras_api
h

?kernel
@bias
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
h

Akernel
Bbias
|	variables
}trainable_variables
~regularization_losses
	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Ckernel
Dbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Ekernel
Fbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Gkernel
Hbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Ikernel
Jbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Kkernel
Lbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Mkernel
Nbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Okernel
Pbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Qkernel
Rbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
30
41
52
63
74
85
96
:7
;8
<9
=10
>11
?12
@13
A14
B15
C16
D17
E18
F19
G20
H21
I22
J23
K24
L25
M26
N27
O28
P29
Q30
R31
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
 	variables
!trainable_variables
"regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
$	variables
%trainable_variables
&regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1

(0
)1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
*	variables
+trainable_variables
,regularization_losses
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
OM
VARIABLE_VALUEblock1_conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock1_conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock1_conv2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock1_conv2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock2_conv1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock2_conv2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock3_conv1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock3_conv1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock3_conv2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock3_conv3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv4/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock3_conv4/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv1/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv1/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv2/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv2/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv3/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv3/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv4/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv4/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv1/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv1/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv2/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv2/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv3/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv3/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv4/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv4/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
?
30
41
52
63
74
85
96
:7
;8
<9
=10
>11
?12
@13
A14
B15
C16
D17
E18
F19
G20
H21
I22
J23
K24
L25
M26
N27
O28
P29
Q30
R31

0
1
2

?0
?1
 
 

30
41
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses

50
61
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
\	variables
]trainable_variables
^regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
`	variables
atrainable_variables
bregularization_losses

70
81
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
d	variables
etrainable_variables
fregularization_losses

90
:1
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
h	variables
itrainable_variables
jregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses

;0
<1
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
p	variables
qtrainable_variables
rregularization_losses

=0
>1
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
t	variables
utrainable_variables
vregularization_losses

?0
@1
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
x	variables
ytrainable_variables
zregularization_losses

A0
B1
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
|	variables
}trainable_variables
~regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

C0
D1
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

E0
F1
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

G0
H1
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

I0
J1
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

K0
L1
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

M0
N1
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

O0
P1
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

Q0
R1
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?
30
41
52
63
74
85
96
:7
;8
<9
=10
>11
?12
@13
A14
B15
C16
D17
E18
F19
G20
H21
I22
J23
K24
L25
M26
N27
O28
P29
Q30
R31
?

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
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
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api

30
41
 
 
 
 

50
61
 
 
 
 
 
 
 
 
 

70
81
 
 
 
 

90
:1
 
 
 
 
 
 
 
 
 

;0
<1
 
 
 
 

=0
>1
 
 
 
 

?0
@1
 
 
 
 

A0
B1
 
 
 
 
 
 
 
 
 

C0
D1
 
 
 
 

E0
F1
 
 
 
 

G0
H1
 
 
 
 

I0
J1
 
 
 
 
 
 
 
 
 

K0
L1
 
 
 
 

M0
N1
 
 
 
 

O0
P1
 
 
 
 

Q0
R1
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_vgg19_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_vgg19_inputblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock4_conv4/kernelblock4_conv4/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasblock5_conv4/kernelblock5_conv4/biasdense/kernel
dense/bias*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_39327
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block3_conv4/kernel/Read/ReadVariableOp%block3_conv4/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block4_conv4/kernel/Read/ReadVariableOp%block4_conv4/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp'block5_conv2/kernel/Read/ReadVariableOp%block5_conv2/bias/Read/ReadVariableOp'block5_conv3/kernel/Read/ReadVariableOp%block5_conv3/bias/Read/ReadVariableOp'block5_conv4/kernel/Read/ReadVariableOp%block5_conv4/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*<
Tin5
321	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_40728
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock4_conv4/kernelblock4_conv4/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasblock5_conv4/kernelblock5_conv4/biastotalcounttotal_1count_1Adam/dense/kernel/mAdam/dense/bias/mAdam/dense/kernel/vAdam/dense/bias/v*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_40879??
?
?
G__inference_block2_conv1_layer_call_and_return_conditional_losses_37639

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????@@?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
,__inference_block5_conv2_layer_call_fn_40493

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_37844x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?	
#__inference_signature_wrapper_39327
vgg19_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?&

unknown_25:??

unknown_26:	?&

unknown_27:??

unknown_28:	?&

unknown_29:??

unknown_30:	?

unknown_31:	?@

unknown_32:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallvgg19_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_37521o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_namevgg19_input
?
?
,__inference_block2_conv1_layer_call_fn_40213

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_37639x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
G__inference_block2_conv1_layer_call_and_return_conditional_losses_40224

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????@@?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
G
+__inference_block2_pool_layer_call_fn_40249

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_37542?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?f
?
@__inference_vgg19_layer_call_and_return_conditional_losses_38617
input_1,
block1_conv1_38531:@ 
block1_conv1_38533:@,
block1_conv2_38536:@@ 
block1_conv2_38538:@-
block2_conv1_38542:@?!
block2_conv1_38544:	?.
block2_conv2_38547:??!
block2_conv2_38549:	?.
block3_conv1_38553:??!
block3_conv1_38555:	?.
block3_conv2_38558:??!
block3_conv2_38560:	?.
block3_conv3_38563:??!
block3_conv3_38565:	?.
block3_conv4_38568:??!
block3_conv4_38570:	?.
block4_conv1_38574:??!
block4_conv1_38576:	?.
block4_conv2_38579:??!
block4_conv2_38581:	?.
block4_conv3_38584:??!
block4_conv3_38586:	?.
block4_conv4_38589:??!
block4_conv4_38591:	?.
block5_conv1_38595:??!
block5_conv1_38597:	?.
block5_conv2_38600:??!
block5_conv2_38602:	?.
block5_conv3_38605:??!
block5_conv3_38607:	?.
block5_conv4_38610:??!
block5_conv4_38612:	?
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block3_conv4/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block4_conv4/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?$block5_conv4/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_38531block1_conv1_38533*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_37599?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_38536block1_conv2_38538*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_37616?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_37626?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_38542block2_conv1_38544*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_37639?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_38547block2_conv2_38549*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_37656?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_37666?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_38553block3_conv1_38555*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_37679?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_38558block3_conv2_38560*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_37696?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_38563block3_conv3_38565*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_37713?
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_38568block3_conv4_38570*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv4_layer_call_and_return_conditional_losses_37730?
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_37740?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_38574block4_conv1_38576*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_37753?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_38579block4_conv2_38581*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_37770?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_38584block4_conv3_38586*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_37787?
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_38589block4_conv4_38591*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv4_layer_call_and_return_conditional_losses_37804?
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_37814?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_38595block5_conv1_38597*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_37827?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_38600block5_conv2_38602*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_37844?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_38605block5_conv3_38607*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_37861?
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_38610block5_conv4_38612*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv4_layer_call_and_return_conditional_losses_37878?
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_37888|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
,__inference_block3_conv2_layer_call_fn_40293

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_37696x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
,__inference_block5_conv1_layer_call_fn_40473

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_37827x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_38708

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
b
F__inference_block1_pool_layer_call_and_return_conditional_losses_37626

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@@@*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
G__inference_block4_conv2_layer_call_and_return_conditional_losses_37770

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?f
?
@__inference_vgg19_layer_call_and_return_conditional_losses_38528
input_1,
block1_conv1_38442:@ 
block1_conv1_38444:@,
block1_conv2_38447:@@ 
block1_conv2_38449:@-
block2_conv1_38453:@?!
block2_conv1_38455:	?.
block2_conv2_38458:??!
block2_conv2_38460:	?.
block3_conv1_38464:??!
block3_conv1_38466:	?.
block3_conv2_38469:??!
block3_conv2_38471:	?.
block3_conv3_38474:??!
block3_conv3_38476:	?.
block3_conv4_38479:??!
block3_conv4_38481:	?.
block4_conv1_38485:??!
block4_conv1_38487:	?.
block4_conv2_38490:??!
block4_conv2_38492:	?.
block4_conv3_38495:??!
block4_conv3_38497:	?.
block4_conv4_38500:??!
block4_conv4_38502:	?.
block5_conv1_38506:??!
block5_conv1_38508:	?.
block5_conv2_38511:??!
block5_conv2_38513:	?.
block5_conv3_38516:??!
block5_conv3_38518:	?.
block5_conv4_38521:??!
block5_conv4_38523:	?
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block3_conv4/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block4_conv4/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?$block5_conv4/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_38442block1_conv1_38444*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_37599?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_38447block1_conv2_38449*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_37616?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_37626?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_38453block2_conv1_38455*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_37639?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_38458block2_conv2_38460*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_37656?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_37666?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_38464block3_conv1_38466*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_37679?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_38469block3_conv2_38471*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_37696?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_38474block3_conv3_38476*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_37713?
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_38479block3_conv4_38481*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv4_layer_call_and_return_conditional_losses_37730?
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_37740?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_38485block4_conv1_38487*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_37753?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_38490block4_conv2_38492*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_37770?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_38495block4_conv3_38497*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_37787?
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_38500block4_conv4_38502*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv4_layer_call_and_return_conditional_losses_37804?
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_37814?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_38506block5_conv1_38508*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_37827?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_38511block5_conv2_38513*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_37844?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_38516block5_conv3_38518*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_37861?
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_38521block5_conv4_38523*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv4_layer_call_and_return_conditional_losses_37878?
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_37888|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
G__inference_block1_conv1_layer_call_and_return_conditional_losses_37599

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
Ѿ
?
E__inference_sequential_layer_call_and_return_conditional_losses_39603

inputsK
1vgg19_block1_conv1_conv2d_readvariableop_resource:@@
2vgg19_block1_conv1_biasadd_readvariableop_resource:@K
1vgg19_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg19_block1_conv2_biasadd_readvariableop_resource:@L
1vgg19_block2_conv1_conv2d_readvariableop_resource:@?A
2vgg19_block2_conv1_biasadd_readvariableop_resource:	?M
1vgg19_block2_conv2_conv2d_readvariableop_resource:??A
2vgg19_block2_conv2_biasadd_readvariableop_resource:	?M
1vgg19_block3_conv1_conv2d_readvariableop_resource:??A
2vgg19_block3_conv1_biasadd_readvariableop_resource:	?M
1vgg19_block3_conv2_conv2d_readvariableop_resource:??A
2vgg19_block3_conv2_biasadd_readvariableop_resource:	?M
1vgg19_block3_conv3_conv2d_readvariableop_resource:??A
2vgg19_block3_conv3_biasadd_readvariableop_resource:	?M
1vgg19_block3_conv4_conv2d_readvariableop_resource:??A
2vgg19_block3_conv4_biasadd_readvariableop_resource:	?M
1vgg19_block4_conv1_conv2d_readvariableop_resource:??A
2vgg19_block4_conv1_biasadd_readvariableop_resource:	?M
1vgg19_block4_conv2_conv2d_readvariableop_resource:??A
2vgg19_block4_conv2_biasadd_readvariableop_resource:	?M
1vgg19_block4_conv3_conv2d_readvariableop_resource:??A
2vgg19_block4_conv3_biasadd_readvariableop_resource:	?M
1vgg19_block4_conv4_conv2d_readvariableop_resource:??A
2vgg19_block4_conv4_biasadd_readvariableop_resource:	?M
1vgg19_block5_conv1_conv2d_readvariableop_resource:??A
2vgg19_block5_conv1_biasadd_readvariableop_resource:	?M
1vgg19_block5_conv2_conv2d_readvariableop_resource:??A
2vgg19_block5_conv2_biasadd_readvariableop_resource:	?M
1vgg19_block5_conv3_conv2d_readvariableop_resource:??A
2vgg19_block5_conv3_biasadd_readvariableop_resource:	?M
1vgg19_block5_conv4_conv2d_readvariableop_resource:??A
2vgg19_block5_conv4_biasadd_readvariableop_resource:	?7
$dense_matmul_readvariableop_resource:	?@3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?)vgg19/block1_conv1/BiasAdd/ReadVariableOp?(vgg19/block1_conv1/Conv2D/ReadVariableOp?)vgg19/block1_conv2/BiasAdd/ReadVariableOp?(vgg19/block1_conv2/Conv2D/ReadVariableOp?)vgg19/block2_conv1/BiasAdd/ReadVariableOp?(vgg19/block2_conv1/Conv2D/ReadVariableOp?)vgg19/block2_conv2/BiasAdd/ReadVariableOp?(vgg19/block2_conv2/Conv2D/ReadVariableOp?)vgg19/block3_conv1/BiasAdd/ReadVariableOp?(vgg19/block3_conv1/Conv2D/ReadVariableOp?)vgg19/block3_conv2/BiasAdd/ReadVariableOp?(vgg19/block3_conv2/Conv2D/ReadVariableOp?)vgg19/block3_conv3/BiasAdd/ReadVariableOp?(vgg19/block3_conv3/Conv2D/ReadVariableOp?)vgg19/block3_conv4/BiasAdd/ReadVariableOp?(vgg19/block3_conv4/Conv2D/ReadVariableOp?)vgg19/block4_conv1/BiasAdd/ReadVariableOp?(vgg19/block4_conv1/Conv2D/ReadVariableOp?)vgg19/block4_conv2/BiasAdd/ReadVariableOp?(vgg19/block4_conv2/Conv2D/ReadVariableOp?)vgg19/block4_conv3/BiasAdd/ReadVariableOp?(vgg19/block4_conv3/Conv2D/ReadVariableOp?)vgg19/block4_conv4/BiasAdd/ReadVariableOp?(vgg19/block4_conv4/Conv2D/ReadVariableOp?)vgg19/block5_conv1/BiasAdd/ReadVariableOp?(vgg19/block5_conv1/Conv2D/ReadVariableOp?)vgg19/block5_conv2/BiasAdd/ReadVariableOp?(vgg19/block5_conv2/Conv2D/ReadVariableOp?)vgg19/block5_conv3/BiasAdd/ReadVariableOp?(vgg19/block5_conv3/Conv2D/ReadVariableOp?)vgg19/block5_conv4/BiasAdd/ReadVariableOp?(vgg19/block5_conv4/Conv2D/ReadVariableOp?
(vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
vgg19/block1_conv1/Conv2DConv2Dinputs0vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
)vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
vgg19/block1_conv1/BiasAddBiasAdd"vgg19/block1_conv1/Conv2D:output:01vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@?
vgg19/block1_conv1/ReluRelu#vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
(vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
vgg19/block1_conv2/Conv2DConv2D%vgg19/block1_conv1/Relu:activations:00vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
)vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
vgg19/block1_conv2/BiasAddBiasAdd"vgg19/block1_conv2/Conv2D:output:01vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@?
vgg19/block1_conv2/ReluRelu#vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
vgg19/block1_pool/MaxPoolMaxPool%vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:?????????@@@*
ksize
*
paddingVALID*
strides
?
(vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
vgg19/block2_conv1/Conv2DConv2D"vgg19/block1_pool/MaxPool:output:00vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
?
)vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block2_conv1/BiasAddBiasAdd"vgg19/block2_conv1/Conv2D:output:01vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?
vgg19/block2_conv1/ReluRelu#vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@??
(vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block2_conv2/Conv2DConv2D%vgg19/block2_conv1/Relu:activations:00vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
?
)vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block2_conv2/BiasAddBiasAdd"vgg19/block2_conv2/Conv2D:output:01vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?
vgg19/block2_conv2/ReluRelu#vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@??
vgg19/block2_pool/MaxPoolMaxPool%vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
?
(vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block3_conv1/Conv2DConv2D"vgg19/block2_pool/MaxPool:output:00vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
)vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block3_conv1/BiasAddBiasAdd"vgg19/block3_conv1/Conv2D:output:01vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?
vgg19/block3_conv1/ReluRelu#vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
(vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block3_conv2/Conv2DConv2D%vgg19/block3_conv1/Relu:activations:00vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
)vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block3_conv2/BiasAddBiasAdd"vgg19/block3_conv2/Conv2D:output:01vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?
vgg19/block3_conv2/ReluRelu#vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
(vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block3_conv3/Conv2DConv2D%vgg19/block3_conv2/Relu:activations:00vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
)vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block3_conv3/BiasAddBiasAdd"vgg19/block3_conv3/Conv2D:output:01vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?
vgg19/block3_conv3/ReluRelu#vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
(vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block3_conv4/Conv2DConv2D%vgg19/block3_conv3/Relu:activations:00vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
)vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block3_conv4/BiasAddBiasAdd"vgg19/block3_conv4/Conv2D:output:01vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?
vgg19/block3_conv4/ReluRelu#vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
vgg19/block3_pool/MaxPoolMaxPool%vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
(vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block4_conv1/Conv2DConv2D"vgg19/block3_pool/MaxPool:output:00vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block4_conv1/BiasAddBiasAdd"vgg19/block4_conv1/Conv2D:output:01vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg19/block4_conv1/ReluRelu#vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block4_conv2/Conv2DConv2D%vgg19/block4_conv1/Relu:activations:00vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block4_conv2/BiasAddBiasAdd"vgg19/block4_conv2/Conv2D:output:01vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg19/block4_conv2/ReluRelu#vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block4_conv3/Conv2DConv2D%vgg19/block4_conv2/Relu:activations:00vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block4_conv3/BiasAddBiasAdd"vgg19/block4_conv3/Conv2D:output:01vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg19/block4_conv3/ReluRelu#vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block4_conv4/Conv2DConv2D%vgg19/block4_conv3/Relu:activations:00vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block4_conv4/BiasAddBiasAdd"vgg19/block4_conv4/Conv2D:output:01vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg19/block4_conv4/ReluRelu#vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
vgg19/block4_pool/MaxPoolMaxPool%vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
(vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block5_conv1/Conv2DConv2D"vgg19/block4_pool/MaxPool:output:00vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block5_conv1/BiasAddBiasAdd"vgg19/block5_conv1/Conv2D:output:01vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg19/block5_conv1/ReluRelu#vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block5_conv2/Conv2DConv2D%vgg19/block5_conv1/Relu:activations:00vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block5_conv2/BiasAddBiasAdd"vgg19/block5_conv2/Conv2D:output:01vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg19/block5_conv2/ReluRelu#vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block5_conv3/Conv2DConv2D%vgg19/block5_conv2/Relu:activations:00vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block5_conv3/BiasAddBiasAdd"vgg19/block5_conv3/Conv2D:output:01vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg19/block5_conv3/ReluRelu#vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block5_conv4/Conv2DConv2D%vgg19/block5_conv3/Relu:activations:00vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block5_conv4/BiasAddBiasAdd"vgg19/block5_conv4/Conv2D:output:01vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg19/block5_conv4/ReluRelu#vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
vgg19/block5_pool/MaxPoolMaxPool%vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ?
flatten/ReshapeReshape"vgg19/block5_pool/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????@?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*^vgg19/block1_conv1/BiasAdd/ReadVariableOp)^vgg19/block1_conv1/Conv2D/ReadVariableOp*^vgg19/block1_conv2/BiasAdd/ReadVariableOp)^vgg19/block1_conv2/Conv2D/ReadVariableOp*^vgg19/block2_conv1/BiasAdd/ReadVariableOp)^vgg19/block2_conv1/Conv2D/ReadVariableOp*^vgg19/block2_conv2/BiasAdd/ReadVariableOp)^vgg19/block2_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv1/BiasAdd/ReadVariableOp)^vgg19/block3_conv1/Conv2D/ReadVariableOp*^vgg19/block3_conv2/BiasAdd/ReadVariableOp)^vgg19/block3_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv3/BiasAdd/ReadVariableOp)^vgg19/block3_conv3/Conv2D/ReadVariableOp*^vgg19/block3_conv4/BiasAdd/ReadVariableOp)^vgg19/block3_conv4/Conv2D/ReadVariableOp*^vgg19/block4_conv1/BiasAdd/ReadVariableOp)^vgg19/block4_conv1/Conv2D/ReadVariableOp*^vgg19/block4_conv2/BiasAdd/ReadVariableOp)^vgg19/block4_conv2/Conv2D/ReadVariableOp*^vgg19/block4_conv3/BiasAdd/ReadVariableOp)^vgg19/block4_conv3/Conv2D/ReadVariableOp*^vgg19/block4_conv4/BiasAdd/ReadVariableOp)^vgg19/block4_conv4/Conv2D/ReadVariableOp*^vgg19/block5_conv1/BiasAdd/ReadVariableOp)^vgg19/block5_conv1/Conv2D/ReadVariableOp*^vgg19/block5_conv2/BiasAdd/ReadVariableOp)^vgg19/block5_conv2/Conv2D/ReadVariableOp*^vgg19/block5_conv3/BiasAdd/ReadVariableOp)^vgg19/block5_conv3/Conv2D/ReadVariableOp*^vgg19/block5_conv4/BiasAdd/ReadVariableOp)^vgg19/block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2V
)vgg19/block1_conv1/BiasAdd/ReadVariableOp)vgg19/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv1/Conv2D/ReadVariableOp(vgg19/block1_conv1/Conv2D/ReadVariableOp2V
)vgg19/block1_conv2/BiasAdd/ReadVariableOp)vgg19/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv2/Conv2D/ReadVariableOp(vgg19/block1_conv2/Conv2D/ReadVariableOp2V
)vgg19/block2_conv1/BiasAdd/ReadVariableOp)vgg19/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv1/Conv2D/ReadVariableOp(vgg19/block2_conv1/Conv2D/ReadVariableOp2V
)vgg19/block2_conv2/BiasAdd/ReadVariableOp)vgg19/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv2/Conv2D/ReadVariableOp(vgg19/block2_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv1/BiasAdd/ReadVariableOp)vgg19/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv1/Conv2D/ReadVariableOp(vgg19/block3_conv1/Conv2D/ReadVariableOp2V
)vgg19/block3_conv2/BiasAdd/ReadVariableOp)vgg19/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv2/Conv2D/ReadVariableOp(vgg19/block3_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv3/BiasAdd/ReadVariableOp)vgg19/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv3/Conv2D/ReadVariableOp(vgg19/block3_conv3/Conv2D/ReadVariableOp2V
)vgg19/block3_conv4/BiasAdd/ReadVariableOp)vgg19/block3_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv4/Conv2D/ReadVariableOp(vgg19/block3_conv4/Conv2D/ReadVariableOp2V
)vgg19/block4_conv1/BiasAdd/ReadVariableOp)vgg19/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv1/Conv2D/ReadVariableOp(vgg19/block4_conv1/Conv2D/ReadVariableOp2V
)vgg19/block4_conv2/BiasAdd/ReadVariableOp)vgg19/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv2/Conv2D/ReadVariableOp(vgg19/block4_conv2/Conv2D/ReadVariableOp2V
)vgg19/block4_conv3/BiasAdd/ReadVariableOp)vgg19/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv3/Conv2D/ReadVariableOp(vgg19/block4_conv3/Conv2D/ReadVariableOp2V
)vgg19/block4_conv4/BiasAdd/ReadVariableOp)vgg19/block4_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv4/Conv2D/ReadVariableOp(vgg19/block4_conv4/Conv2D/ReadVariableOp2V
)vgg19/block5_conv1/BiasAdd/ReadVariableOp)vgg19/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv1/Conv2D/ReadVariableOp(vgg19/block5_conv1/Conv2D/ReadVariableOp2V
)vgg19/block5_conv2/BiasAdd/ReadVariableOp)vgg19/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv2/Conv2D/ReadVariableOp(vgg19/block5_conv2/Conv2D/ReadVariableOp2V
)vgg19/block5_conv3/BiasAdd/ReadVariableOp)vgg19/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv3/Conv2D/ReadVariableOp(vgg19/block5_conv3/Conv2D/ReadVariableOp2V
)vgg19/block5_conv4/BiasAdd/ReadVariableOp)vgg19/block5_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv4/Conv2D/ReadVariableOp(vgg19/block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_block3_pool_layer_call_and_return_conditional_losses_40364

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?

E__inference_sequential_layer_call_and_return_conditional_losses_39171
vgg19_input%
vgg19_39099:@
vgg19_39101:@%
vgg19_39103:@@
vgg19_39105:@&
vgg19_39107:@?
vgg19_39109:	?'
vgg19_39111:??
vgg19_39113:	?'
vgg19_39115:??
vgg19_39117:	?'
vgg19_39119:??
vgg19_39121:	?'
vgg19_39123:??
vgg19_39125:	?'
vgg19_39127:??
vgg19_39129:	?'
vgg19_39131:??
vgg19_39133:	?'
vgg19_39135:??
vgg19_39137:	?'
vgg19_39139:??
vgg19_39141:	?'
vgg19_39143:??
vgg19_39145:	?'
vgg19_39147:??
vgg19_39149:	?'
vgg19_39151:??
vgg19_39153:	?'
vgg19_39155:??
vgg19_39157:	?'
vgg19_39159:??
vgg19_39161:	?
dense_39165:	?@
dense_39167:
identity??dense/StatefulPartitionedCall?vgg19/StatefulPartitionedCall?
vgg19/StatefulPartitionedCallStatefulPartitionedCallvgg19_inputvgg19_39099vgg19_39101vgg19_39103vgg19_39105vgg19_39107vgg19_39109vgg19_39111vgg19_39113vgg19_39115vgg19_39117vgg19_39119vgg19_39121vgg19_39123vgg19_39125vgg19_39127vgg19_39129vgg19_39131vgg19_39133vgg19_39135vgg19_39137vgg19_39139vgg19_39141vgg19_39143vgg19_39145vgg19_39147vgg19_39149vgg19_39151vgg19_39153vgg19_39155vgg19_39157vgg19_39159vgg19_39161*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg19_layer_call_and_return_conditional_losses_37891?
flatten/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_38695?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_39165dense_39167*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_38708u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_namevgg19_input
?
?
G__inference_block3_conv2_layer_call_and_return_conditional_losses_40304

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????  ?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????  ?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
%__inference_dense_layer_call_fn_40133

inputs
unknown:	?@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_38708o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
G__inference_block2_conv2_layer_call_and_return_conditional_losses_37656

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????@@?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
G__inference_block5_conv4_layer_call_and_return_conditional_losses_37878

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?$
 __inference__wrapped_model_37521
vgg19_inputV
<sequential_vgg19_block1_conv1_conv2d_readvariableop_resource:@K
=sequential_vgg19_block1_conv1_biasadd_readvariableop_resource:@V
<sequential_vgg19_block1_conv2_conv2d_readvariableop_resource:@@K
=sequential_vgg19_block1_conv2_biasadd_readvariableop_resource:@W
<sequential_vgg19_block2_conv1_conv2d_readvariableop_resource:@?L
=sequential_vgg19_block2_conv1_biasadd_readvariableop_resource:	?X
<sequential_vgg19_block2_conv2_conv2d_readvariableop_resource:??L
=sequential_vgg19_block2_conv2_biasadd_readvariableop_resource:	?X
<sequential_vgg19_block3_conv1_conv2d_readvariableop_resource:??L
=sequential_vgg19_block3_conv1_biasadd_readvariableop_resource:	?X
<sequential_vgg19_block3_conv2_conv2d_readvariableop_resource:??L
=sequential_vgg19_block3_conv2_biasadd_readvariableop_resource:	?X
<sequential_vgg19_block3_conv3_conv2d_readvariableop_resource:??L
=sequential_vgg19_block3_conv3_biasadd_readvariableop_resource:	?X
<sequential_vgg19_block3_conv4_conv2d_readvariableop_resource:??L
=sequential_vgg19_block3_conv4_biasadd_readvariableop_resource:	?X
<sequential_vgg19_block4_conv1_conv2d_readvariableop_resource:??L
=sequential_vgg19_block4_conv1_biasadd_readvariableop_resource:	?X
<sequential_vgg19_block4_conv2_conv2d_readvariableop_resource:??L
=sequential_vgg19_block4_conv2_biasadd_readvariableop_resource:	?X
<sequential_vgg19_block4_conv3_conv2d_readvariableop_resource:??L
=sequential_vgg19_block4_conv3_biasadd_readvariableop_resource:	?X
<sequential_vgg19_block4_conv4_conv2d_readvariableop_resource:??L
=sequential_vgg19_block4_conv4_biasadd_readvariableop_resource:	?X
<sequential_vgg19_block5_conv1_conv2d_readvariableop_resource:??L
=sequential_vgg19_block5_conv1_biasadd_readvariableop_resource:	?X
<sequential_vgg19_block5_conv2_conv2d_readvariableop_resource:??L
=sequential_vgg19_block5_conv2_biasadd_readvariableop_resource:	?X
<sequential_vgg19_block5_conv3_conv2d_readvariableop_resource:??L
=sequential_vgg19_block5_conv3_biasadd_readvariableop_resource:	?X
<sequential_vgg19_block5_conv4_conv2d_readvariableop_resource:??L
=sequential_vgg19_block5_conv4_biasadd_readvariableop_resource:	?B
/sequential_dense_matmul_readvariableop_resource:	?@>
0sequential_dense_biasadd_readvariableop_resource:
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?4sequential/vgg19/block1_conv1/BiasAdd/ReadVariableOp?3sequential/vgg19/block1_conv1/Conv2D/ReadVariableOp?4sequential/vgg19/block1_conv2/BiasAdd/ReadVariableOp?3sequential/vgg19/block1_conv2/Conv2D/ReadVariableOp?4sequential/vgg19/block2_conv1/BiasAdd/ReadVariableOp?3sequential/vgg19/block2_conv1/Conv2D/ReadVariableOp?4sequential/vgg19/block2_conv2/BiasAdd/ReadVariableOp?3sequential/vgg19/block2_conv2/Conv2D/ReadVariableOp?4sequential/vgg19/block3_conv1/BiasAdd/ReadVariableOp?3sequential/vgg19/block3_conv1/Conv2D/ReadVariableOp?4sequential/vgg19/block3_conv2/BiasAdd/ReadVariableOp?3sequential/vgg19/block3_conv2/Conv2D/ReadVariableOp?4sequential/vgg19/block3_conv3/BiasAdd/ReadVariableOp?3sequential/vgg19/block3_conv3/Conv2D/ReadVariableOp?4sequential/vgg19/block3_conv4/BiasAdd/ReadVariableOp?3sequential/vgg19/block3_conv4/Conv2D/ReadVariableOp?4sequential/vgg19/block4_conv1/BiasAdd/ReadVariableOp?3sequential/vgg19/block4_conv1/Conv2D/ReadVariableOp?4sequential/vgg19/block4_conv2/BiasAdd/ReadVariableOp?3sequential/vgg19/block4_conv2/Conv2D/ReadVariableOp?4sequential/vgg19/block4_conv3/BiasAdd/ReadVariableOp?3sequential/vgg19/block4_conv3/Conv2D/ReadVariableOp?4sequential/vgg19/block4_conv4/BiasAdd/ReadVariableOp?3sequential/vgg19/block4_conv4/Conv2D/ReadVariableOp?4sequential/vgg19/block5_conv1/BiasAdd/ReadVariableOp?3sequential/vgg19/block5_conv1/Conv2D/ReadVariableOp?4sequential/vgg19/block5_conv2/BiasAdd/ReadVariableOp?3sequential/vgg19/block5_conv2/Conv2D/ReadVariableOp?4sequential/vgg19/block5_conv3/BiasAdd/ReadVariableOp?3sequential/vgg19/block5_conv3/Conv2D/ReadVariableOp?4sequential/vgg19/block5_conv4/BiasAdd/ReadVariableOp?3sequential/vgg19/block5_conv4/Conv2D/ReadVariableOp?
3sequential/vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
$sequential/vgg19/block1_conv1/Conv2DConv2Dvgg19_input;sequential/vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
4sequential/vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
%sequential/vgg19/block1_conv1/BiasAddBiasAdd-sequential/vgg19/block1_conv1/Conv2D:output:0<sequential/vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@?
"sequential/vgg19/block1_conv1/ReluRelu.sequential/vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
3sequential/vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
$sequential/vgg19/block1_conv2/Conv2DConv2D0sequential/vgg19/block1_conv1/Relu:activations:0;sequential/vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
4sequential/vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
%sequential/vgg19/block1_conv2/BiasAddBiasAdd-sequential/vgg19/block1_conv2/Conv2D:output:0<sequential/vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@?
"sequential/vgg19/block1_conv2/ReluRelu.sequential/vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
$sequential/vgg19/block1_pool/MaxPoolMaxPool0sequential/vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:?????????@@@*
ksize
*
paddingVALID*
strides
?
3sequential/vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
$sequential/vgg19/block2_conv1/Conv2DConv2D-sequential/vgg19/block1_pool/MaxPool:output:0;sequential/vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
?
4sequential/vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%sequential/vgg19/block2_conv1/BiasAddBiasAdd-sequential/vgg19/block2_conv1/Conv2D:output:0<sequential/vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@??
"sequential/vgg19/block2_conv1/ReluRelu.sequential/vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@??
3sequential/vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
$sequential/vgg19/block2_conv2/Conv2DConv2D0sequential/vgg19/block2_conv1/Relu:activations:0;sequential/vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
?
4sequential/vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%sequential/vgg19/block2_conv2/BiasAddBiasAdd-sequential/vgg19/block2_conv2/Conv2D:output:0<sequential/vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@??
"sequential/vgg19/block2_conv2/ReluRelu.sequential/vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@??
$sequential/vgg19/block2_pool/MaxPoolMaxPool0sequential/vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
?
3sequential/vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
$sequential/vgg19/block3_conv1/Conv2DConv2D-sequential/vgg19/block2_pool/MaxPool:output:0;sequential/vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
4sequential/vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%sequential/vgg19/block3_conv1/BiasAddBiasAdd-sequential/vgg19/block3_conv1/Conv2D:output:0<sequential/vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ??
"sequential/vgg19/block3_conv1/ReluRelu.sequential/vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
3sequential/vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
$sequential/vgg19/block3_conv2/Conv2DConv2D0sequential/vgg19/block3_conv1/Relu:activations:0;sequential/vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
4sequential/vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%sequential/vgg19/block3_conv2/BiasAddBiasAdd-sequential/vgg19/block3_conv2/Conv2D:output:0<sequential/vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ??
"sequential/vgg19/block3_conv2/ReluRelu.sequential/vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
3sequential/vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
$sequential/vgg19/block3_conv3/Conv2DConv2D0sequential/vgg19/block3_conv2/Relu:activations:0;sequential/vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
4sequential/vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%sequential/vgg19/block3_conv3/BiasAddBiasAdd-sequential/vgg19/block3_conv3/Conv2D:output:0<sequential/vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ??
"sequential/vgg19/block3_conv3/ReluRelu.sequential/vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
3sequential/vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
$sequential/vgg19/block3_conv4/Conv2DConv2D0sequential/vgg19/block3_conv3/Relu:activations:0;sequential/vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
4sequential/vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%sequential/vgg19/block3_conv4/BiasAddBiasAdd-sequential/vgg19/block3_conv4/Conv2D:output:0<sequential/vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ??
"sequential/vgg19/block3_conv4/ReluRelu.sequential/vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
$sequential/vgg19/block3_pool/MaxPoolMaxPool0sequential/vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
3sequential/vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
$sequential/vgg19/block4_conv1/Conv2DConv2D-sequential/vgg19/block3_pool/MaxPool:output:0;sequential/vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
4sequential/vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%sequential/vgg19/block4_conv1/BiasAddBiasAdd-sequential/vgg19/block4_conv1/Conv2D:output:0<sequential/vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
"sequential/vgg19/block4_conv1/ReluRelu.sequential/vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
3sequential/vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
$sequential/vgg19/block4_conv2/Conv2DConv2D0sequential/vgg19/block4_conv1/Relu:activations:0;sequential/vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
4sequential/vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%sequential/vgg19/block4_conv2/BiasAddBiasAdd-sequential/vgg19/block4_conv2/Conv2D:output:0<sequential/vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
"sequential/vgg19/block4_conv2/ReluRelu.sequential/vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
3sequential/vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
$sequential/vgg19/block4_conv3/Conv2DConv2D0sequential/vgg19/block4_conv2/Relu:activations:0;sequential/vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
4sequential/vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%sequential/vgg19/block4_conv3/BiasAddBiasAdd-sequential/vgg19/block4_conv3/Conv2D:output:0<sequential/vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
"sequential/vgg19/block4_conv3/ReluRelu.sequential/vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
3sequential/vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
$sequential/vgg19/block4_conv4/Conv2DConv2D0sequential/vgg19/block4_conv3/Relu:activations:0;sequential/vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
4sequential/vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%sequential/vgg19/block4_conv4/BiasAddBiasAdd-sequential/vgg19/block4_conv4/Conv2D:output:0<sequential/vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
"sequential/vgg19/block4_conv4/ReluRelu.sequential/vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
$sequential/vgg19/block4_pool/MaxPoolMaxPool0sequential/vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
3sequential/vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
$sequential/vgg19/block5_conv1/Conv2DConv2D-sequential/vgg19/block4_pool/MaxPool:output:0;sequential/vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
4sequential/vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%sequential/vgg19/block5_conv1/BiasAddBiasAdd-sequential/vgg19/block5_conv1/Conv2D:output:0<sequential/vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
"sequential/vgg19/block5_conv1/ReluRelu.sequential/vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
3sequential/vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
$sequential/vgg19/block5_conv2/Conv2DConv2D0sequential/vgg19/block5_conv1/Relu:activations:0;sequential/vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
4sequential/vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%sequential/vgg19/block5_conv2/BiasAddBiasAdd-sequential/vgg19/block5_conv2/Conv2D:output:0<sequential/vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
"sequential/vgg19/block5_conv2/ReluRelu.sequential/vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
3sequential/vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
$sequential/vgg19/block5_conv3/Conv2DConv2D0sequential/vgg19/block5_conv2/Relu:activations:0;sequential/vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
4sequential/vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%sequential/vgg19/block5_conv3/BiasAddBiasAdd-sequential/vgg19/block5_conv3/Conv2D:output:0<sequential/vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
"sequential/vgg19/block5_conv3/ReluRelu.sequential/vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
3sequential/vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
$sequential/vgg19/block5_conv4/Conv2DConv2D0sequential/vgg19/block5_conv3/Relu:activations:0;sequential/vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
4sequential/vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%sequential/vgg19/block5_conv4/BiasAddBiasAdd-sequential/vgg19/block5_conv4/Conv2D:output:0<sequential/vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
"sequential/vgg19/block5_conv4/ReluRelu.sequential/vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
$sequential/vgg19/block5_pool/MaxPoolMaxPool0sequential/vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ?
sequential/flatten/ReshapeReshape-sequential/vgg19/block5_pool/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????@?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x
sequential/dense/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????k
IdentityIdentitysequential/dense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp5^sequential/vgg19/block1_conv1/BiasAdd/ReadVariableOp4^sequential/vgg19/block1_conv1/Conv2D/ReadVariableOp5^sequential/vgg19/block1_conv2/BiasAdd/ReadVariableOp4^sequential/vgg19/block1_conv2/Conv2D/ReadVariableOp5^sequential/vgg19/block2_conv1/BiasAdd/ReadVariableOp4^sequential/vgg19/block2_conv1/Conv2D/ReadVariableOp5^sequential/vgg19/block2_conv2/BiasAdd/ReadVariableOp4^sequential/vgg19/block2_conv2/Conv2D/ReadVariableOp5^sequential/vgg19/block3_conv1/BiasAdd/ReadVariableOp4^sequential/vgg19/block3_conv1/Conv2D/ReadVariableOp5^sequential/vgg19/block3_conv2/BiasAdd/ReadVariableOp4^sequential/vgg19/block3_conv2/Conv2D/ReadVariableOp5^sequential/vgg19/block3_conv3/BiasAdd/ReadVariableOp4^sequential/vgg19/block3_conv3/Conv2D/ReadVariableOp5^sequential/vgg19/block3_conv4/BiasAdd/ReadVariableOp4^sequential/vgg19/block3_conv4/Conv2D/ReadVariableOp5^sequential/vgg19/block4_conv1/BiasAdd/ReadVariableOp4^sequential/vgg19/block4_conv1/Conv2D/ReadVariableOp5^sequential/vgg19/block4_conv2/BiasAdd/ReadVariableOp4^sequential/vgg19/block4_conv2/Conv2D/ReadVariableOp5^sequential/vgg19/block4_conv3/BiasAdd/ReadVariableOp4^sequential/vgg19/block4_conv3/Conv2D/ReadVariableOp5^sequential/vgg19/block4_conv4/BiasAdd/ReadVariableOp4^sequential/vgg19/block4_conv4/Conv2D/ReadVariableOp5^sequential/vgg19/block5_conv1/BiasAdd/ReadVariableOp4^sequential/vgg19/block5_conv1/Conv2D/ReadVariableOp5^sequential/vgg19/block5_conv2/BiasAdd/ReadVariableOp4^sequential/vgg19/block5_conv2/Conv2D/ReadVariableOp5^sequential/vgg19/block5_conv3/BiasAdd/ReadVariableOp4^sequential/vgg19/block5_conv3/Conv2D/ReadVariableOp5^sequential/vgg19/block5_conv4/BiasAdd/ReadVariableOp4^sequential/vgg19/block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2l
4sequential/vgg19/block1_conv1/BiasAdd/ReadVariableOp4sequential/vgg19/block1_conv1/BiasAdd/ReadVariableOp2j
3sequential/vgg19/block1_conv1/Conv2D/ReadVariableOp3sequential/vgg19/block1_conv1/Conv2D/ReadVariableOp2l
4sequential/vgg19/block1_conv2/BiasAdd/ReadVariableOp4sequential/vgg19/block1_conv2/BiasAdd/ReadVariableOp2j
3sequential/vgg19/block1_conv2/Conv2D/ReadVariableOp3sequential/vgg19/block1_conv2/Conv2D/ReadVariableOp2l
4sequential/vgg19/block2_conv1/BiasAdd/ReadVariableOp4sequential/vgg19/block2_conv1/BiasAdd/ReadVariableOp2j
3sequential/vgg19/block2_conv1/Conv2D/ReadVariableOp3sequential/vgg19/block2_conv1/Conv2D/ReadVariableOp2l
4sequential/vgg19/block2_conv2/BiasAdd/ReadVariableOp4sequential/vgg19/block2_conv2/BiasAdd/ReadVariableOp2j
3sequential/vgg19/block2_conv2/Conv2D/ReadVariableOp3sequential/vgg19/block2_conv2/Conv2D/ReadVariableOp2l
4sequential/vgg19/block3_conv1/BiasAdd/ReadVariableOp4sequential/vgg19/block3_conv1/BiasAdd/ReadVariableOp2j
3sequential/vgg19/block3_conv1/Conv2D/ReadVariableOp3sequential/vgg19/block3_conv1/Conv2D/ReadVariableOp2l
4sequential/vgg19/block3_conv2/BiasAdd/ReadVariableOp4sequential/vgg19/block3_conv2/BiasAdd/ReadVariableOp2j
3sequential/vgg19/block3_conv2/Conv2D/ReadVariableOp3sequential/vgg19/block3_conv2/Conv2D/ReadVariableOp2l
4sequential/vgg19/block3_conv3/BiasAdd/ReadVariableOp4sequential/vgg19/block3_conv3/BiasAdd/ReadVariableOp2j
3sequential/vgg19/block3_conv3/Conv2D/ReadVariableOp3sequential/vgg19/block3_conv3/Conv2D/ReadVariableOp2l
4sequential/vgg19/block3_conv4/BiasAdd/ReadVariableOp4sequential/vgg19/block3_conv4/BiasAdd/ReadVariableOp2j
3sequential/vgg19/block3_conv4/Conv2D/ReadVariableOp3sequential/vgg19/block3_conv4/Conv2D/ReadVariableOp2l
4sequential/vgg19/block4_conv1/BiasAdd/ReadVariableOp4sequential/vgg19/block4_conv1/BiasAdd/ReadVariableOp2j
3sequential/vgg19/block4_conv1/Conv2D/ReadVariableOp3sequential/vgg19/block4_conv1/Conv2D/ReadVariableOp2l
4sequential/vgg19/block4_conv2/BiasAdd/ReadVariableOp4sequential/vgg19/block4_conv2/BiasAdd/ReadVariableOp2j
3sequential/vgg19/block4_conv2/Conv2D/ReadVariableOp3sequential/vgg19/block4_conv2/Conv2D/ReadVariableOp2l
4sequential/vgg19/block4_conv3/BiasAdd/ReadVariableOp4sequential/vgg19/block4_conv3/BiasAdd/ReadVariableOp2j
3sequential/vgg19/block4_conv3/Conv2D/ReadVariableOp3sequential/vgg19/block4_conv3/Conv2D/ReadVariableOp2l
4sequential/vgg19/block4_conv4/BiasAdd/ReadVariableOp4sequential/vgg19/block4_conv4/BiasAdd/ReadVariableOp2j
3sequential/vgg19/block4_conv4/Conv2D/ReadVariableOp3sequential/vgg19/block4_conv4/Conv2D/ReadVariableOp2l
4sequential/vgg19/block5_conv1/BiasAdd/ReadVariableOp4sequential/vgg19/block5_conv1/BiasAdd/ReadVariableOp2j
3sequential/vgg19/block5_conv1/Conv2D/ReadVariableOp3sequential/vgg19/block5_conv1/Conv2D/ReadVariableOp2l
4sequential/vgg19/block5_conv2/BiasAdd/ReadVariableOp4sequential/vgg19/block5_conv2/BiasAdd/ReadVariableOp2j
3sequential/vgg19/block5_conv2/Conv2D/ReadVariableOp3sequential/vgg19/block5_conv2/Conv2D/ReadVariableOp2l
4sequential/vgg19/block5_conv3/BiasAdd/ReadVariableOp4sequential/vgg19/block5_conv3/BiasAdd/ReadVariableOp2j
3sequential/vgg19/block5_conv3/Conv2D/ReadVariableOp3sequential/vgg19/block5_conv3/Conv2D/ReadVariableOp2l
4sequential/vgg19/block5_conv4/BiasAdd/ReadVariableOp4sequential/vgg19/block5_conv4/BiasAdd/ReadVariableOp2j
3sequential/vgg19/block5_conv4/Conv2D/ReadVariableOp3sequential/vgg19/block5_conv4/Conv2D/ReadVariableOp:^ Z
1
_output_shapes
:???????????
%
_user_specified_namevgg19_input
?
?
%__inference_vgg19_layer_call_fn_39871

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?&

unknown_25:??

unknown_26:	?&

unknown_27:??

unknown_28:	?&

unknown_29:??

unknown_30:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg19_layer_call_and_return_conditional_losses_38303x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
@__inference_vgg19_layer_call_and_return_conditional_losses_40113

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@?;
,block2_conv1_biasadd_readvariableop_resource:	?G
+block2_conv2_conv2d_readvariableop_resource:??;
,block2_conv2_biasadd_readvariableop_resource:	?G
+block3_conv1_conv2d_readvariableop_resource:??;
,block3_conv1_biasadd_readvariableop_resource:	?G
+block3_conv2_conv2d_readvariableop_resource:??;
,block3_conv2_biasadd_readvariableop_resource:	?G
+block3_conv3_conv2d_readvariableop_resource:??;
,block3_conv3_biasadd_readvariableop_resource:	?G
+block3_conv4_conv2d_readvariableop_resource:??;
,block3_conv4_biasadd_readvariableop_resource:	?G
+block4_conv1_conv2d_readvariableop_resource:??;
,block4_conv1_biasadd_readvariableop_resource:	?G
+block4_conv2_conv2d_readvariableop_resource:??;
,block4_conv2_biasadd_readvariableop_resource:	?G
+block4_conv3_conv2d_readvariableop_resource:??;
,block4_conv3_biasadd_readvariableop_resource:	?G
+block4_conv4_conv2d_readvariableop_resource:??;
,block4_conv4_biasadd_readvariableop_resource:	?G
+block5_conv1_conv2d_readvariableop_resource:??;
,block5_conv1_biasadd_readvariableop_resource:	?G
+block5_conv2_conv2d_readvariableop_resource:??;
,block5_conv2_biasadd_readvariableop_resource:	?G
+block5_conv3_conv2d_readvariableop_resource:??;
,block5_conv3_biasadd_readvariableop_resource:	?G
+block5_conv4_conv2d_readvariableop_resource:??;
,block5_conv4_biasadd_readvariableop_resource:	?
identity??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?#block3_conv2/BiasAdd/ReadVariableOp?"block3_conv2/Conv2D/ReadVariableOp?#block3_conv3/BiasAdd/ReadVariableOp?"block3_conv3/Conv2D/ReadVariableOp?#block3_conv4/BiasAdd/ReadVariableOp?"block3_conv4/Conv2D/ReadVariableOp?#block4_conv1/BiasAdd/ReadVariableOp?"block4_conv1/Conv2D/ReadVariableOp?#block4_conv2/BiasAdd/ReadVariableOp?"block4_conv2/Conv2D/ReadVariableOp?#block4_conv3/BiasAdd/ReadVariableOp?"block4_conv3/Conv2D/ReadVariableOp?#block4_conv4/BiasAdd/ReadVariableOp?"block4_conv4/Conv2D/ReadVariableOp?#block5_conv1/BiasAdd/ReadVariableOp?"block5_conv1/Conv2D/ReadVariableOp?#block5_conv2/BiasAdd/ReadVariableOp?"block5_conv2/Conv2D/ReadVariableOp?#block5_conv3/BiasAdd/ReadVariableOp?"block5_conv3/Conv2D/ReadVariableOp?#block5_conv4/BiasAdd/ReadVariableOp?"block5_conv4/Conv2D/ReadVariableOp?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:?????????@@@*
ksize
*
paddingVALID*
strides
?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?s
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@??
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?s
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@??
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?s
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
t
IdentityIdentityblock5_pool/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:???????????	
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_40124

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????@Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
F__inference_block4_pool_layer_call_and_return_conditional_losses_37814

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?	
*__inference_sequential_layer_call_fn_39096
vgg19_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?&

unknown_25:??

unknown_26:	?&

unknown_27:??

unknown_28:	?&

unknown_29:??

unknown_30:	?

unknown_31:	?@

unknown_32:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallvgg19_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_38952o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_namevgg19_input
?
?
G__inference_block5_conv3_layer_call_and_return_conditional_losses_40524

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block1_conv2_layer_call_and_return_conditional_losses_40184

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
b
F__inference_block5_pool_layer_call_and_return_conditional_losses_40559

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_block4_conv2_layer_call_and_return_conditional_losses_40404

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_38695

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????@Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_block4_conv4_layer_call_fn_40433

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv4_layer_call_and_return_conditional_losses_37804x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_block5_conv4_layer_call_fn_40533

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv4_layer_call_and_return_conditional_losses_37878x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_block2_pool_layer_call_fn_40254

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_37666i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  ?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?

E__inference_sequential_layer_call_and_return_conditional_losses_39246
vgg19_input%
vgg19_39174:@
vgg19_39176:@%
vgg19_39178:@@
vgg19_39180:@&
vgg19_39182:@?
vgg19_39184:	?'
vgg19_39186:??
vgg19_39188:	?'
vgg19_39190:??
vgg19_39192:	?'
vgg19_39194:??
vgg19_39196:	?'
vgg19_39198:??
vgg19_39200:	?'
vgg19_39202:??
vgg19_39204:	?'
vgg19_39206:??
vgg19_39208:	?'
vgg19_39210:??
vgg19_39212:	?'
vgg19_39214:??
vgg19_39216:	?'
vgg19_39218:??
vgg19_39220:	?'
vgg19_39222:??
vgg19_39224:	?'
vgg19_39226:??
vgg19_39228:	?'
vgg19_39230:??
vgg19_39232:	?'
vgg19_39234:??
vgg19_39236:	?
dense_39240:	?@
dense_39242:
identity??dense/StatefulPartitionedCall?vgg19/StatefulPartitionedCall?
vgg19/StatefulPartitionedCallStatefulPartitionedCallvgg19_inputvgg19_39174vgg19_39176vgg19_39178vgg19_39180vgg19_39182vgg19_39184vgg19_39186vgg19_39188vgg19_39190vgg19_39192vgg19_39194vgg19_39196vgg19_39198vgg19_39200vgg19_39202vgg19_39204vgg19_39206vgg19_39208vgg19_39210vgg19_39212vgg19_39214vgg19_39216vgg19_39218vgg19_39220vgg19_39222vgg19_39224vgg19_39226vgg19_39228vgg19_39230vgg19_39232vgg19_39234vgg19_39236*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg19_layer_call_and_return_conditional_losses_38303?
flatten/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_38695?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_39240dense_39242*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_38708u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_namevgg19_input
?
?
,__inference_block4_conv1_layer_call_fn_40373

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_37753x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_block5_conv3_layer_call_fn_40513

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_37861x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block5_conv2_layer_call_and_return_conditional_losses_40504

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_block3_pool_layer_call_fn_40354

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_37740i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
b
F__inference_block4_pool_layer_call_and_return_conditional_losses_40464

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?f
?
@__inference_vgg19_layer_call_and_return_conditional_losses_38303

inputs,
block1_conv1_38217:@ 
block1_conv1_38219:@,
block1_conv2_38222:@@ 
block1_conv2_38224:@-
block2_conv1_38228:@?!
block2_conv1_38230:	?.
block2_conv2_38233:??!
block2_conv2_38235:	?.
block3_conv1_38239:??!
block3_conv1_38241:	?.
block3_conv2_38244:??!
block3_conv2_38246:	?.
block3_conv3_38249:??!
block3_conv3_38251:	?.
block3_conv4_38254:??!
block3_conv4_38256:	?.
block4_conv1_38260:??!
block4_conv1_38262:	?.
block4_conv2_38265:??!
block4_conv2_38267:	?.
block4_conv3_38270:??!
block4_conv3_38272:	?.
block4_conv4_38275:??!
block4_conv4_38277:	?.
block5_conv1_38281:??!
block5_conv1_38283:	?.
block5_conv2_38286:??!
block5_conv2_38288:	?.
block5_conv3_38291:??!
block5_conv3_38293:	?.
block5_conv4_38296:??!
block5_conv4_38298:	?
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block3_conv4/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block4_conv4/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?$block5_conv4/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_38217block1_conv1_38219*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_37599?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_38222block1_conv2_38224*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_37616?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_37626?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_38228block2_conv1_38230*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_37639?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_38233block2_conv2_38235*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_37656?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_37666?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_38239block3_conv1_38241*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_37679?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_38244block3_conv2_38246*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_37696?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_38249block3_conv3_38251*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_37713?
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_38254block3_conv4_38256*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv4_layer_call_and_return_conditional_losses_37730?
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_37740?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_38260block4_conv1_38262*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_37753?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_38265block4_conv2_38267*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_37770?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_38270block4_conv3_38272*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_37787?
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_38275block4_conv4_38277*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv4_layer_call_and_return_conditional_losses_37804?
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_37814?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_38281block5_conv1_38283*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_37827?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_38286block5_conv2_38288*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_37844?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_38291block5_conv3_38293*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_37861?
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_38296block5_conv4_38298*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv4_layer_call_and_return_conditional_losses_37878?
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_37888|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
,__inference_block4_conv2_layer_call_fn_40393

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_37770x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_block3_pool_layer_call_fn_40349

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_37554?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_block4_pool_layer_call_and_return_conditional_losses_37566

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_block1_pool_layer_call_and_return_conditional_losses_40199

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
Ѿ
?
E__inference_sequential_layer_call_and_return_conditional_losses_39733

inputsK
1vgg19_block1_conv1_conv2d_readvariableop_resource:@@
2vgg19_block1_conv1_biasadd_readvariableop_resource:@K
1vgg19_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg19_block1_conv2_biasadd_readvariableop_resource:@L
1vgg19_block2_conv1_conv2d_readvariableop_resource:@?A
2vgg19_block2_conv1_biasadd_readvariableop_resource:	?M
1vgg19_block2_conv2_conv2d_readvariableop_resource:??A
2vgg19_block2_conv2_biasadd_readvariableop_resource:	?M
1vgg19_block3_conv1_conv2d_readvariableop_resource:??A
2vgg19_block3_conv1_biasadd_readvariableop_resource:	?M
1vgg19_block3_conv2_conv2d_readvariableop_resource:??A
2vgg19_block3_conv2_biasadd_readvariableop_resource:	?M
1vgg19_block3_conv3_conv2d_readvariableop_resource:??A
2vgg19_block3_conv3_biasadd_readvariableop_resource:	?M
1vgg19_block3_conv4_conv2d_readvariableop_resource:??A
2vgg19_block3_conv4_biasadd_readvariableop_resource:	?M
1vgg19_block4_conv1_conv2d_readvariableop_resource:??A
2vgg19_block4_conv1_biasadd_readvariableop_resource:	?M
1vgg19_block4_conv2_conv2d_readvariableop_resource:??A
2vgg19_block4_conv2_biasadd_readvariableop_resource:	?M
1vgg19_block4_conv3_conv2d_readvariableop_resource:??A
2vgg19_block4_conv3_biasadd_readvariableop_resource:	?M
1vgg19_block4_conv4_conv2d_readvariableop_resource:??A
2vgg19_block4_conv4_biasadd_readvariableop_resource:	?M
1vgg19_block5_conv1_conv2d_readvariableop_resource:??A
2vgg19_block5_conv1_biasadd_readvariableop_resource:	?M
1vgg19_block5_conv2_conv2d_readvariableop_resource:??A
2vgg19_block5_conv2_biasadd_readvariableop_resource:	?M
1vgg19_block5_conv3_conv2d_readvariableop_resource:??A
2vgg19_block5_conv3_biasadd_readvariableop_resource:	?M
1vgg19_block5_conv4_conv2d_readvariableop_resource:??A
2vgg19_block5_conv4_biasadd_readvariableop_resource:	?7
$dense_matmul_readvariableop_resource:	?@3
%dense_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?)vgg19/block1_conv1/BiasAdd/ReadVariableOp?(vgg19/block1_conv1/Conv2D/ReadVariableOp?)vgg19/block1_conv2/BiasAdd/ReadVariableOp?(vgg19/block1_conv2/Conv2D/ReadVariableOp?)vgg19/block2_conv1/BiasAdd/ReadVariableOp?(vgg19/block2_conv1/Conv2D/ReadVariableOp?)vgg19/block2_conv2/BiasAdd/ReadVariableOp?(vgg19/block2_conv2/Conv2D/ReadVariableOp?)vgg19/block3_conv1/BiasAdd/ReadVariableOp?(vgg19/block3_conv1/Conv2D/ReadVariableOp?)vgg19/block3_conv2/BiasAdd/ReadVariableOp?(vgg19/block3_conv2/Conv2D/ReadVariableOp?)vgg19/block3_conv3/BiasAdd/ReadVariableOp?(vgg19/block3_conv3/Conv2D/ReadVariableOp?)vgg19/block3_conv4/BiasAdd/ReadVariableOp?(vgg19/block3_conv4/Conv2D/ReadVariableOp?)vgg19/block4_conv1/BiasAdd/ReadVariableOp?(vgg19/block4_conv1/Conv2D/ReadVariableOp?)vgg19/block4_conv2/BiasAdd/ReadVariableOp?(vgg19/block4_conv2/Conv2D/ReadVariableOp?)vgg19/block4_conv3/BiasAdd/ReadVariableOp?(vgg19/block4_conv3/Conv2D/ReadVariableOp?)vgg19/block4_conv4/BiasAdd/ReadVariableOp?(vgg19/block4_conv4/Conv2D/ReadVariableOp?)vgg19/block5_conv1/BiasAdd/ReadVariableOp?(vgg19/block5_conv1/Conv2D/ReadVariableOp?)vgg19/block5_conv2/BiasAdd/ReadVariableOp?(vgg19/block5_conv2/Conv2D/ReadVariableOp?)vgg19/block5_conv3/BiasAdd/ReadVariableOp?(vgg19/block5_conv3/Conv2D/ReadVariableOp?)vgg19/block5_conv4/BiasAdd/ReadVariableOp?(vgg19/block5_conv4/Conv2D/ReadVariableOp?
(vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
vgg19/block1_conv1/Conv2DConv2Dinputs0vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
)vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
vgg19/block1_conv1/BiasAddBiasAdd"vgg19/block1_conv1/Conv2D:output:01vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@?
vgg19/block1_conv1/ReluRelu#vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
(vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
vgg19/block1_conv2/Conv2DConv2D%vgg19/block1_conv1/Relu:activations:00vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
)vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
vgg19/block1_conv2/BiasAddBiasAdd"vgg19/block1_conv2/Conv2D:output:01vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@?
vgg19/block1_conv2/ReluRelu#vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
vgg19/block1_pool/MaxPoolMaxPool%vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:?????????@@@*
ksize
*
paddingVALID*
strides
?
(vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
vgg19/block2_conv1/Conv2DConv2D"vgg19/block1_pool/MaxPool:output:00vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
?
)vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block2_conv1/BiasAddBiasAdd"vgg19/block2_conv1/Conv2D:output:01vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?
vgg19/block2_conv1/ReluRelu#vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@??
(vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block2_conv2/Conv2DConv2D%vgg19/block2_conv1/Relu:activations:00vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
?
)vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block2_conv2/BiasAddBiasAdd"vgg19/block2_conv2/Conv2D:output:01vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?
vgg19/block2_conv2/ReluRelu#vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@??
vgg19/block2_pool/MaxPoolMaxPool%vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
?
(vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block3_conv1/Conv2DConv2D"vgg19/block2_pool/MaxPool:output:00vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
)vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block3_conv1/BiasAddBiasAdd"vgg19/block3_conv1/Conv2D:output:01vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?
vgg19/block3_conv1/ReluRelu#vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
(vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block3_conv2/Conv2DConv2D%vgg19/block3_conv1/Relu:activations:00vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
)vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block3_conv2/BiasAddBiasAdd"vgg19/block3_conv2/Conv2D:output:01vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?
vgg19/block3_conv2/ReluRelu#vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
(vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block3_conv3/Conv2DConv2D%vgg19/block3_conv2/Relu:activations:00vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
)vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block3_conv3/BiasAddBiasAdd"vgg19/block3_conv3/Conv2D:output:01vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?
vgg19/block3_conv3/ReluRelu#vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
(vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block3_conv4/Conv2DConv2D%vgg19/block3_conv3/Relu:activations:00vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
)vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block3_conv4/BiasAddBiasAdd"vgg19/block3_conv4/Conv2D:output:01vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?
vgg19/block3_conv4/ReluRelu#vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
vgg19/block3_pool/MaxPoolMaxPool%vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
(vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block4_conv1/Conv2DConv2D"vgg19/block3_pool/MaxPool:output:00vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block4_conv1/BiasAddBiasAdd"vgg19/block4_conv1/Conv2D:output:01vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg19/block4_conv1/ReluRelu#vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block4_conv2/Conv2DConv2D%vgg19/block4_conv1/Relu:activations:00vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block4_conv2/BiasAddBiasAdd"vgg19/block4_conv2/Conv2D:output:01vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg19/block4_conv2/ReluRelu#vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block4_conv3/Conv2DConv2D%vgg19/block4_conv2/Relu:activations:00vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block4_conv3/BiasAddBiasAdd"vgg19/block4_conv3/Conv2D:output:01vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg19/block4_conv3/ReluRelu#vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block4_conv4/Conv2DConv2D%vgg19/block4_conv3/Relu:activations:00vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block4_conv4/BiasAddBiasAdd"vgg19/block4_conv4/Conv2D:output:01vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg19/block4_conv4/ReluRelu#vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
vgg19/block4_pool/MaxPoolMaxPool%vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
(vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block5_conv1/Conv2DConv2D"vgg19/block4_pool/MaxPool:output:00vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block5_conv1/BiasAddBiasAdd"vgg19/block5_conv1/Conv2D:output:01vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg19/block5_conv1/ReluRelu#vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block5_conv2/Conv2DConv2D%vgg19/block5_conv1/Relu:activations:00vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block5_conv2/BiasAddBiasAdd"vgg19/block5_conv2/Conv2D:output:01vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg19/block5_conv2/ReluRelu#vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block5_conv3/Conv2DConv2D%vgg19/block5_conv2/Relu:activations:00vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block5_conv3/BiasAddBiasAdd"vgg19/block5_conv3/Conv2D:output:01vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg19/block5_conv3/ReluRelu#vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg19/block5_conv4/Conv2DConv2D%vgg19/block5_conv3/Relu:activations:00vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg19/block5_conv4/BiasAddBiasAdd"vgg19/block5_conv4/Conv2D:output:01vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg19/block5_conv4/ReluRelu#vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
vgg19/block5_pool/MaxPoolMaxPool%vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ?
flatten/ReshapeReshape"vgg19/block5_pool/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????@?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*^vgg19/block1_conv1/BiasAdd/ReadVariableOp)^vgg19/block1_conv1/Conv2D/ReadVariableOp*^vgg19/block1_conv2/BiasAdd/ReadVariableOp)^vgg19/block1_conv2/Conv2D/ReadVariableOp*^vgg19/block2_conv1/BiasAdd/ReadVariableOp)^vgg19/block2_conv1/Conv2D/ReadVariableOp*^vgg19/block2_conv2/BiasAdd/ReadVariableOp)^vgg19/block2_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv1/BiasAdd/ReadVariableOp)^vgg19/block3_conv1/Conv2D/ReadVariableOp*^vgg19/block3_conv2/BiasAdd/ReadVariableOp)^vgg19/block3_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv3/BiasAdd/ReadVariableOp)^vgg19/block3_conv3/Conv2D/ReadVariableOp*^vgg19/block3_conv4/BiasAdd/ReadVariableOp)^vgg19/block3_conv4/Conv2D/ReadVariableOp*^vgg19/block4_conv1/BiasAdd/ReadVariableOp)^vgg19/block4_conv1/Conv2D/ReadVariableOp*^vgg19/block4_conv2/BiasAdd/ReadVariableOp)^vgg19/block4_conv2/Conv2D/ReadVariableOp*^vgg19/block4_conv3/BiasAdd/ReadVariableOp)^vgg19/block4_conv3/Conv2D/ReadVariableOp*^vgg19/block4_conv4/BiasAdd/ReadVariableOp)^vgg19/block4_conv4/Conv2D/ReadVariableOp*^vgg19/block5_conv1/BiasAdd/ReadVariableOp)^vgg19/block5_conv1/Conv2D/ReadVariableOp*^vgg19/block5_conv2/BiasAdd/ReadVariableOp)^vgg19/block5_conv2/Conv2D/ReadVariableOp*^vgg19/block5_conv3/BiasAdd/ReadVariableOp)^vgg19/block5_conv3/Conv2D/ReadVariableOp*^vgg19/block5_conv4/BiasAdd/ReadVariableOp)^vgg19/block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2V
)vgg19/block1_conv1/BiasAdd/ReadVariableOp)vgg19/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv1/Conv2D/ReadVariableOp(vgg19/block1_conv1/Conv2D/ReadVariableOp2V
)vgg19/block1_conv2/BiasAdd/ReadVariableOp)vgg19/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv2/Conv2D/ReadVariableOp(vgg19/block1_conv2/Conv2D/ReadVariableOp2V
)vgg19/block2_conv1/BiasAdd/ReadVariableOp)vgg19/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv1/Conv2D/ReadVariableOp(vgg19/block2_conv1/Conv2D/ReadVariableOp2V
)vgg19/block2_conv2/BiasAdd/ReadVariableOp)vgg19/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv2/Conv2D/ReadVariableOp(vgg19/block2_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv1/BiasAdd/ReadVariableOp)vgg19/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv1/Conv2D/ReadVariableOp(vgg19/block3_conv1/Conv2D/ReadVariableOp2V
)vgg19/block3_conv2/BiasAdd/ReadVariableOp)vgg19/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv2/Conv2D/ReadVariableOp(vgg19/block3_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv3/BiasAdd/ReadVariableOp)vgg19/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv3/Conv2D/ReadVariableOp(vgg19/block3_conv3/Conv2D/ReadVariableOp2V
)vgg19/block3_conv4/BiasAdd/ReadVariableOp)vgg19/block3_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv4/Conv2D/ReadVariableOp(vgg19/block3_conv4/Conv2D/ReadVariableOp2V
)vgg19/block4_conv1/BiasAdd/ReadVariableOp)vgg19/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv1/Conv2D/ReadVariableOp(vgg19/block4_conv1/Conv2D/ReadVariableOp2V
)vgg19/block4_conv2/BiasAdd/ReadVariableOp)vgg19/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv2/Conv2D/ReadVariableOp(vgg19/block4_conv2/Conv2D/ReadVariableOp2V
)vgg19/block4_conv3/BiasAdd/ReadVariableOp)vgg19/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv3/Conv2D/ReadVariableOp(vgg19/block4_conv3/Conv2D/ReadVariableOp2V
)vgg19/block4_conv4/BiasAdd/ReadVariableOp)vgg19/block4_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv4/Conv2D/ReadVariableOp(vgg19/block4_conv4/Conv2D/ReadVariableOp2V
)vgg19/block5_conv1/BiasAdd/ReadVariableOp)vgg19/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv1/Conv2D/ReadVariableOp(vgg19/block5_conv1/Conv2D/ReadVariableOp2V
)vgg19/block5_conv2/BiasAdd/ReadVariableOp)vgg19/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv2/Conv2D/ReadVariableOp(vgg19/block5_conv2/Conv2D/ReadVariableOp2V
)vgg19/block5_conv3/BiasAdd/ReadVariableOp)vgg19/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv3/Conv2D/ReadVariableOp(vgg19/block5_conv3/Conv2D/ReadVariableOp2V
)vgg19/block5_conv4/BiasAdd/ReadVariableOp)vgg19/block5_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv4/Conv2D/ReadVariableOp(vgg19/block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_block1_pool_layer_call_and_return_conditional_losses_40204

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@@@*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?f
?
@__inference_vgg19_layer_call_and_return_conditional_losses_37891

inputs,
block1_conv1_37600:@ 
block1_conv1_37602:@,
block1_conv2_37617:@@ 
block1_conv2_37619:@-
block2_conv1_37640:@?!
block2_conv1_37642:	?.
block2_conv2_37657:??!
block2_conv2_37659:	?.
block3_conv1_37680:??!
block3_conv1_37682:	?.
block3_conv2_37697:??!
block3_conv2_37699:	?.
block3_conv3_37714:??!
block3_conv3_37716:	?.
block3_conv4_37731:??!
block3_conv4_37733:	?.
block4_conv1_37754:??!
block4_conv1_37756:	?.
block4_conv2_37771:??!
block4_conv2_37773:	?.
block4_conv3_37788:??!
block4_conv3_37790:	?.
block4_conv4_37805:??!
block4_conv4_37807:	?.
block5_conv1_37828:??!
block5_conv1_37830:	?.
block5_conv2_37845:??!
block5_conv2_37847:	?.
block5_conv3_37862:??!
block5_conv3_37864:	?.
block5_conv4_37879:??!
block5_conv4_37881:	?
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block3_conv4/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block4_conv4/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?$block5_conv4/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_37600block1_conv1_37602*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_37599?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_37617block1_conv2_37619*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_37616?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_37626?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_37640block2_conv1_37642*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_37639?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_37657block2_conv2_37659*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_37656?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_37666?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_37680block3_conv1_37682*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_37679?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_37697block3_conv2_37699*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_37696?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_37714block3_conv3_37716*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_37713?
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_37731block3_conv4_37733*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv4_layer_call_and_return_conditional_losses_37730?
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_37740?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_37754block4_conv1_37756*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_37753?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_37771block4_conv2_37773*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_37770?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_37788block4_conv3_37790*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_37787?
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_37805block4_conv4_37807*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv4_layer_call_and_return_conditional_losses_37804?
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_37814?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_37828block5_conv1_37830*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_37827?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_37845block5_conv2_37847*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_37844?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_37862block5_conv3_37864*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_37861?
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_37879block5_conv4_37881*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv4_layer_call_and_return_conditional_losses_37878?
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_37888|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
G__inference_block5_conv1_layer_call_and_return_conditional_losses_37827

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block4_conv4_layer_call_and_return_conditional_losses_37804

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?	
*__inference_sequential_layer_call_fn_39473

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?&

unknown_25:??

unknown_26:	?&

unknown_27:??

unknown_28:	?&

unknown_29:??

unknown_30:	?

unknown_31:	?@

unknown_32:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_38952o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
G__inference_block5_conv4_layer_call_and_return_conditional_losses_40544

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
%__inference_vgg19_layer_call_fn_37958
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?&

unknown_25:??

unknown_26:	?&

unknown_27:??

unknown_28:	?&

unknown_29:??

unknown_30:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg19_layer_call_and_return_conditional_losses_37891x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
G
+__inference_block1_pool_layer_call_fn_40194

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_37626h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
b
F__inference_block3_pool_layer_call_and_return_conditional_losses_40359

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_block3_conv3_layer_call_fn_40313

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_37713x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
G__inference_block4_conv3_layer_call_and_return_conditional_losses_37787

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block5_conv3_layer_call_and_return_conditional_losses_37861

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block3_conv3_layer_call_and_return_conditional_losses_37713

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????  ?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????  ?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
G__inference_block5_conv2_layer_call_and_return_conditional_losses_37844

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block1_conv2_layer_call_and_return_conditional_losses_37616

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
b
F__inference_block2_pool_layer_call_and_return_conditional_losses_37542

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_block5_pool_layer_call_and_return_conditional_losses_40564

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_block5_pool_layer_call_fn_40554

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_37888i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block3_conv1_layer_call_and_return_conditional_losses_37679

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????  ?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????  ?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?

E__inference_sequential_layer_call_and_return_conditional_losses_38952

inputs%
vgg19_38880:@
vgg19_38882:@%
vgg19_38884:@@
vgg19_38886:@&
vgg19_38888:@?
vgg19_38890:	?'
vgg19_38892:??
vgg19_38894:	?'
vgg19_38896:??
vgg19_38898:	?'
vgg19_38900:??
vgg19_38902:	?'
vgg19_38904:??
vgg19_38906:	?'
vgg19_38908:??
vgg19_38910:	?'
vgg19_38912:??
vgg19_38914:	?'
vgg19_38916:??
vgg19_38918:	?'
vgg19_38920:??
vgg19_38922:	?'
vgg19_38924:??
vgg19_38926:	?'
vgg19_38928:??
vgg19_38930:	?'
vgg19_38932:??
vgg19_38934:	?'
vgg19_38936:??
vgg19_38938:	?'
vgg19_38940:??
vgg19_38942:	?
dense_38946:	?@
dense_38948:
identity??dense/StatefulPartitionedCall?vgg19/StatefulPartitionedCall?
vgg19/StatefulPartitionedCallStatefulPartitionedCallinputsvgg19_38880vgg19_38882vgg19_38884vgg19_38886vgg19_38888vgg19_38890vgg19_38892vgg19_38894vgg19_38896vgg19_38898vgg19_38900vgg19_38902vgg19_38904vgg19_38906vgg19_38908vgg19_38910vgg19_38912vgg19_38914vgg19_38916vgg19_38918vgg19_38920vgg19_38922vgg19_38924vgg19_38926vgg19_38928vgg19_38930vgg19_38932vgg19_38934vgg19_38936vgg19_38938vgg19_38940vgg19_38942*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg19_layer_call_and_return_conditional_losses_38303?
flatten/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_38695?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_38946dense_38948*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_38708u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_block4_pool_layer_call_and_return_conditional_losses_40459

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_block2_conv2_layer_call_and_return_conditional_losses_40244

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????@@?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
%__inference_vgg19_layer_call_fn_38439
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?&

unknown_25:??

unknown_26:	?&

unknown_27:??

unknown_28:	?&

unknown_29:??

unknown_30:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg19_layer_call_and_return_conditional_losses_38303x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
,__inference_block1_conv1_layer_call_fn_40153

inputs!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_37599y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
G__inference_block1_conv1_layer_call_and_return_conditional_losses_40164

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
G__inference_block4_conv4_layer_call_and_return_conditional_losses_40444

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_block1_conv2_layer_call_fn_40173

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_37616y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?

E__inference_sequential_layer_call_and_return_conditional_losses_38715

inputs%
vgg19_38624:@
vgg19_38626:@%
vgg19_38628:@@
vgg19_38630:@&
vgg19_38632:@?
vgg19_38634:	?'
vgg19_38636:??
vgg19_38638:	?'
vgg19_38640:??
vgg19_38642:	?'
vgg19_38644:??
vgg19_38646:	?'
vgg19_38648:??
vgg19_38650:	?'
vgg19_38652:??
vgg19_38654:	?'
vgg19_38656:??
vgg19_38658:	?'
vgg19_38660:??
vgg19_38662:	?'
vgg19_38664:??
vgg19_38666:	?'
vgg19_38668:??
vgg19_38670:	?'
vgg19_38672:??
vgg19_38674:	?'
vgg19_38676:??
vgg19_38678:	?'
vgg19_38680:??
vgg19_38682:	?'
vgg19_38684:??
vgg19_38686:	?
dense_38709:	?@
dense_38711:
identity??dense/StatefulPartitionedCall?vgg19/StatefulPartitionedCall?
vgg19/StatefulPartitionedCallStatefulPartitionedCallinputsvgg19_38624vgg19_38626vgg19_38628vgg19_38630vgg19_38632vgg19_38634vgg19_38636vgg19_38638vgg19_38640vgg19_38642vgg19_38644vgg19_38646vgg19_38648vgg19_38650vgg19_38652vgg19_38654vgg19_38656vgg19_38658vgg19_38660vgg19_38662vgg19_38664vgg19_38666vgg19_38668vgg19_38670vgg19_38672vgg19_38674vgg19_38676vgg19_38678vgg19_38680vgg19_38682vgg19_38684vgg19_38686*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg19_layer_call_and_return_conditional_losses_37891?
flatten/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_38695?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_38709dense_38711*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_38708u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_block2_pool_layer_call_and_return_conditional_losses_40264

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:?????????  ?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
b
F__inference_block2_pool_layer_call_and_return_conditional_losses_37666

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:?????????  ?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
G__inference_block3_conv2_layer_call_and_return_conditional_losses_37696

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????  ?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????  ?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?	
*__inference_sequential_layer_call_fn_38786
vgg19_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?&

unknown_25:??

unknown_26:	?&

unknown_27:??

unknown_28:	?&

unknown_29:??

unknown_30:	?

unknown_31:	?@

unknown_32:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallvgg19_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_38715o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_namevgg19_input
?
?
,__inference_block3_conv4_layer_call_fn_40333

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv4_layer_call_and_return_conditional_losses_37730x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
G__inference_block3_conv4_layer_call_and_return_conditional_losses_40344

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????  ?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????  ?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
G
+__inference_block1_pool_layer_call_fn_40189

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_37530?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_block3_conv1_layer_call_fn_40273

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_37679x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
G__inference_block4_conv3_layer_call_and_return_conditional_losses_40424

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
F__inference_block1_pool_layer_call_and_return_conditional_losses_37530

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
%__inference_vgg19_layer_call_fn_39802

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?&

unknown_25:??

unknown_26:	?&

unknown_27:??

unknown_28:	?&

unknown_29:??

unknown_30:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg19_layer_call_and_return_conditional_losses_37891x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
G__inference_block3_conv1_layer_call_and_return_conditional_losses_40284

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????  ?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????  ?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
G__inference_block3_conv3_layer_call_and_return_conditional_losses_40324

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????  ?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????  ?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
G__inference_block4_conv1_layer_call_and_return_conditional_losses_40384

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_block4_pool_layer_call_fn_40449

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_37566?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?	
*__inference_sequential_layer_call_fn_39400

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?&

unknown_25:??

unknown_26:	?&

unknown_27:??

unknown_28:	?&

unknown_29:??

unknown_30:	?

unknown_31:	?@

unknown_32:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_38715o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_block5_pool_layer_call_and_return_conditional_losses_37888

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_block4_pool_layer_call_fn_40454

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_37814i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
F__inference_block3_pool_layer_call_and_return_conditional_losses_37554

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_block5_pool_layer_call_fn_40549

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_37578?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_block5_conv1_layer_call_and_return_conditional_losses_40484

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_40118

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_38695a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_40144

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
b
F__inference_block3_pool_layer_call_and_return_conditional_losses_37740

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
,__inference_block4_conv3_layer_call_fn_40413

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_37787x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
F__inference_block2_pool_layer_call_and_return_conditional_losses_40259

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_block2_conv2_layer_call_fn_40233

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_37656x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
G__inference_block4_conv1_layer_call_and_return_conditional_losses_37753

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block3_conv4_layer_call_and_return_conditional_losses_37730

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????  ?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????  ?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
??
?
@__inference_vgg19_layer_call_and_return_conditional_losses_39992

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@?;
,block2_conv1_biasadd_readvariableop_resource:	?G
+block2_conv2_conv2d_readvariableop_resource:??;
,block2_conv2_biasadd_readvariableop_resource:	?G
+block3_conv1_conv2d_readvariableop_resource:??;
,block3_conv1_biasadd_readvariableop_resource:	?G
+block3_conv2_conv2d_readvariableop_resource:??;
,block3_conv2_biasadd_readvariableop_resource:	?G
+block3_conv3_conv2d_readvariableop_resource:??;
,block3_conv3_biasadd_readvariableop_resource:	?G
+block3_conv4_conv2d_readvariableop_resource:??;
,block3_conv4_biasadd_readvariableop_resource:	?G
+block4_conv1_conv2d_readvariableop_resource:??;
,block4_conv1_biasadd_readvariableop_resource:	?G
+block4_conv2_conv2d_readvariableop_resource:??;
,block4_conv2_biasadd_readvariableop_resource:	?G
+block4_conv3_conv2d_readvariableop_resource:??;
,block4_conv3_biasadd_readvariableop_resource:	?G
+block4_conv4_conv2d_readvariableop_resource:??;
,block4_conv4_biasadd_readvariableop_resource:	?G
+block5_conv1_conv2d_readvariableop_resource:??;
,block5_conv1_biasadd_readvariableop_resource:	?G
+block5_conv2_conv2d_readvariableop_resource:??;
,block5_conv2_biasadd_readvariableop_resource:	?G
+block5_conv3_conv2d_readvariableop_resource:??;
,block5_conv3_biasadd_readvariableop_resource:	?G
+block5_conv4_conv2d_readvariableop_resource:??;
,block5_conv4_biasadd_readvariableop_resource:	?
identity??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?#block3_conv2/BiasAdd/ReadVariableOp?"block3_conv2/Conv2D/ReadVariableOp?#block3_conv3/BiasAdd/ReadVariableOp?"block3_conv3/Conv2D/ReadVariableOp?#block3_conv4/BiasAdd/ReadVariableOp?"block3_conv4/Conv2D/ReadVariableOp?#block4_conv1/BiasAdd/ReadVariableOp?"block4_conv1/Conv2D/ReadVariableOp?#block4_conv2/BiasAdd/ReadVariableOp?"block4_conv2/Conv2D/ReadVariableOp?#block4_conv3/BiasAdd/ReadVariableOp?"block4_conv3/Conv2D/ReadVariableOp?#block4_conv4/BiasAdd/ReadVariableOp?"block4_conv4/Conv2D/ReadVariableOp?#block5_conv1/BiasAdd/ReadVariableOp?"block5_conv1/Conv2D/ReadVariableOp?#block5_conv2/BiasAdd/ReadVariableOp?"block5_conv2/Conv2D/ReadVariableOp?#block5_conv3/BiasAdd/ReadVariableOp?"block5_conv3/Conv2D/ReadVariableOp?#block5_conv4/BiasAdd/ReadVariableOp?"block5_conv4/Conv2D/ReadVariableOp?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:?????????@@@*
ksize
*
paddingVALID*
strides
?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?s
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@??
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?s
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@??
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?s
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ??
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
t
IdentityIdentityblock5_pool/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:???????????	
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_block5_pool_layer_call_and_return_conditional_losses_37578

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?W
?
__inference__traced_save_40728
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block3_conv4_kernel_read_readvariableop0
,savev2_block3_conv4_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop2
.savev2_block4_conv3_kernel_read_readvariableop0
,savev2_block4_conv3_bias_read_readvariableop2
.savev2_block4_conv4_kernel_read_readvariableop0
,savev2_block4_conv4_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop2
.savev2_block5_conv2_kernel_read_readvariableop0
,savev2_block5_conv2_bias_read_readvariableop2
.savev2_block5_conv3_kernel_read_readvariableop0
,savev2_block5_conv3_bias_read_readvariableop2
.savev2_block5_conv4_kernel_read_readvariableop0
,savev2_block5_conv4_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*?
value?B?0B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block3_conv4_kernel_read_readvariableop,savev2_block3_conv4_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block4_conv4_kernel_read_readvariableop,savev2_block4_conv4_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop.savev2_block5_conv3_kernel_read_readvariableop,savev2_block5_conv3_bias_read_readvariableop.savev2_block5_conv4_kernel_read_readvariableop,savev2_block5_conv4_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *>
dtypes4
220	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?@:: : : : : :@:@:@@:@:@?:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?: : : : :	?@::	?@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 	

_output_shapes
:@:,
(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:. *
(
_output_shapes
:??:!!

_output_shapes	
:?:."*
(
_output_shapes
:??:!#

_output_shapes	
:?:.$*
(
_output_shapes
:??:!%

_output_shapes	
:?:.&*
(
_output_shapes
:??:!'

_output_shapes	
:?:(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :%,!

_output_shapes
:	?@: -

_output_shapes
::%.!

_output_shapes
:	?@: /

_output_shapes
::0

_output_shapes
: 
??
?
!__inference__traced_restore_40879
file_prefix0
assignvariableop_dense_kernel:	?@+
assignvariableop_1_dense_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: @
&assignvariableop_7_block1_conv1_kernel:@2
$assignvariableop_8_block1_conv1_bias:@@
&assignvariableop_9_block1_conv2_kernel:@@3
%assignvariableop_10_block1_conv2_bias:@B
'assignvariableop_11_block2_conv1_kernel:@?4
%assignvariableop_12_block2_conv1_bias:	?C
'assignvariableop_13_block2_conv2_kernel:??4
%assignvariableop_14_block2_conv2_bias:	?C
'assignvariableop_15_block3_conv1_kernel:??4
%assignvariableop_16_block3_conv1_bias:	?C
'assignvariableop_17_block3_conv2_kernel:??4
%assignvariableop_18_block3_conv2_bias:	?C
'assignvariableop_19_block3_conv3_kernel:??4
%assignvariableop_20_block3_conv3_bias:	?C
'assignvariableop_21_block3_conv4_kernel:??4
%assignvariableop_22_block3_conv4_bias:	?C
'assignvariableop_23_block4_conv1_kernel:??4
%assignvariableop_24_block4_conv1_bias:	?C
'assignvariableop_25_block4_conv2_kernel:??4
%assignvariableop_26_block4_conv2_bias:	?C
'assignvariableop_27_block4_conv3_kernel:??4
%assignvariableop_28_block4_conv3_bias:	?C
'assignvariableop_29_block4_conv4_kernel:??4
%assignvariableop_30_block4_conv4_bias:	?C
'assignvariableop_31_block5_conv1_kernel:??4
%assignvariableop_32_block5_conv1_bias:	?C
'assignvariableop_33_block5_conv2_kernel:??4
%assignvariableop_34_block5_conv2_bias:	?C
'assignvariableop_35_block5_conv3_kernel:??4
%assignvariableop_36_block5_conv3_bias:	?C
'assignvariableop_37_block5_conv4_kernel:??4
%assignvariableop_38_block5_conv4_bias:	?#
assignvariableop_39_total: #
assignvariableop_40_count: %
assignvariableop_41_total_1: %
assignvariableop_42_count_1: :
'assignvariableop_43_adam_dense_kernel_m:	?@3
%assignvariableop_44_adam_dense_bias_m::
'assignvariableop_45_adam_dense_kernel_v:	?@3
%assignvariableop_46_adam_dense_bias_v:
identity_48??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*?
value?B?0B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
220	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp&assignvariableop_7_block1_conv1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp$assignvariableop_8_block1_conv1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp&assignvariableop_9_block1_conv2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp%assignvariableop_10_block1_conv2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp'assignvariableop_11_block2_conv1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_block2_conv1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp'assignvariableop_13_block2_conv2_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp%assignvariableop_14_block2_conv2_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_block3_conv1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp%assignvariableop_16_block3_conv1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp'assignvariableop_17_block3_conv2_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_block3_conv2_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp'assignvariableop_19_block3_conv3_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp%assignvariableop_20_block3_conv3_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_block3_conv4_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp%assignvariableop_22_block3_conv4_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_block4_conv1_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_block4_conv1_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_block4_conv2_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp%assignvariableop_26_block4_conv2_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp'assignvariableop_27_block4_conv3_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp%assignvariableop_28_block4_conv3_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_block4_conv4_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp%assignvariableop_30_block4_conv4_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp'assignvariableop_31_block5_conv1_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp%assignvariableop_32_block5_conv1_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp'assignvariableop_33_block5_conv2_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp%assignvariableop_34_block5_conv2_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp'assignvariableop_35_block5_conv3_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp%assignvariableop_36_block5_conv3_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp'assignvariableop_37_block5_conv4_kernelIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp%assignvariableop_38_block5_conv4_biasIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOpassignvariableop_39_totalIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOpassignvariableop_40_countIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOpassignvariableop_41_total_1Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOpassignvariableop_42_count_1Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_dense_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp%assignvariableop_44_adam_dense_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_dense_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp%assignvariableop_46_adam_dense_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_47Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_48IdentityIdentity_47:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_48Identity_48:output:0*s
_input_shapesb
`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_46AssignVariableOp_462(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
M
vgg19_input>
serving_default_vgg19_input:0???????????9
dense0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_sequential
?

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer_with_weights-11
layer-15
layer-16
layer_with_weights-12
layer-17
layer_with_weights-13
layer-18
layer_with_weights-14
layer-19
layer_with_weights-15
layer-20
layer-21
 	variables
!trainable_variables
"regularization_losses
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_network
?
$	variables
%trainable_variables
&regularization_losses
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
{
.iter

/beta_1

0beta_2
	1decay
2learning_rate(m?)m?(v?)v?"
	optimizer
?
30
41
52
63
74
85
96
:7
;8
<9
=10
>11
?12
@13
A14
B15
C16
D17
E18
F19
G20
H21
I22
J23
K24
L25
M26
N27
O28
P29
Q30
R31
(32
)33"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
"
_tf_keras_input_layer
?

3kernel
4bias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

5kernel
6bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
`	variables
atrainable_variables
bregularization_losses
c	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

7kernel
8bias
d	variables
etrainable_variables
fregularization_losses
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

9kernel
:bias
h	variables
itrainable_variables
jregularization_losses
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

;kernel
<bias
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

=kernel
>bias
t	variables
utrainable_variables
vregularization_losses
w	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

?kernel
@bias
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Akernel
Bbias
|	variables
}trainable_variables
~regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ckernel
Dbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ekernel
Fbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Gkernel
Hbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ikernel
Jbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Kkernel
Lbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Mkernel
Nbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Okernel
Pbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Qkernel
Rbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
30
41
52
63
74
85
96
:7
;8
<9
=10
>11
?12
@13
A14
B15
C16
D17
E18
F19
G20
H21
I22
J23
K24
L25
M26
N27
O28
P29
Q30
R31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
 	variables
!trainable_variables
"regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
$	variables
%trainable_variables
&regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	?@2dense/kernel
:2
dense/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
*	variables
+trainable_variables
,regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
.:,@?2block2_conv1/kernel
 :?2block2_conv1/bias
/:-??2block2_conv2/kernel
 :?2block2_conv2/bias
/:-??2block3_conv1/kernel
 :?2block3_conv1/bias
/:-??2block3_conv2/kernel
 :?2block3_conv2/bias
/:-??2block3_conv3/kernel
 :?2block3_conv3/bias
/:-??2block3_conv4/kernel
 :?2block3_conv4/bias
/:-??2block4_conv1/kernel
 :?2block4_conv1/bias
/:-??2block4_conv2/kernel
 :?2block4_conv2/bias
/:-??2block4_conv3/kernel
 :?2block4_conv3/bias
/:-??2block4_conv4/kernel
 :?2block4_conv4/bias
/:-??2block5_conv1/kernel
 :?2block5_conv1/bias
/:-??2block5_conv2/kernel
 :?2block5_conv2/bias
/:-??2block5_conv3/kernel
 :?2block5_conv3/bias
/:-??2block5_conv4/kernel
 :?2block5_conv4/bias
?
30
41
52
63
74
85
96
:7
;8
<9
=10
>11
?12
@13
A14
B15
C16
D17
E18
F19
G20
H21
I22
J23
K24
L25
M26
N27
O28
P29
Q30
R31"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
\	variables
]trainable_variables
^regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
`	variables
atrainable_variables
bregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
d	variables
etrainable_variables
fregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
h	variables
itrainable_variables
jregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
p	variables
qtrainable_variables
rregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
t	variables
utrainable_variables
vregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
x	variables
ytrainable_variables
zregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
|	variables
}trainable_variables
~regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
30
41
52
63
74
85
96
:7
;8
<9
=10
>11
?12
@13
A14
B15
C16
D17
E18
F19
G20
H21
I22
J23
K24
L25
M26
N27
O28
P29
Q30
R31"
trackable_list_wrapper
?

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21"
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
50
61"
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
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
90
:1"
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
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
A0
B1"
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
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
I0
J1"
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
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
Q0
R1"
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
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
$:"	?@2Adam/dense/kernel/m
:2Adam/dense/bias/m
$:"	?@2Adam/dense/kernel/v
:2Adam/dense/bias/v
?2?
*__inference_sequential_layer_call_fn_38786
*__inference_sequential_layer_call_fn_39400
*__inference_sequential_layer_call_fn_39473
*__inference_sequential_layer_call_fn_39096?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_39603
E__inference_sequential_layer_call_and_return_conditional_losses_39733
E__inference_sequential_layer_call_and_return_conditional_losses_39171
E__inference_sequential_layer_call_and_return_conditional_losses_39246?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_37521vgg19_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_vgg19_layer_call_fn_37958
%__inference_vgg19_layer_call_fn_39802
%__inference_vgg19_layer_call_fn_39871
%__inference_vgg19_layer_call_fn_38439?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_vgg19_layer_call_and_return_conditional_losses_39992
@__inference_vgg19_layer_call_and_return_conditional_losses_40113
@__inference_vgg19_layer_call_and_return_conditional_losses_38528
@__inference_vgg19_layer_call_and_return_conditional_losses_38617?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_flatten_layer_call_fn_40118?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_40124?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_40133?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_40144?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_39327vgg19_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block1_conv1_layer_call_fn_40153?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block1_conv1_layer_call_and_return_conditional_losses_40164?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block1_conv2_layer_call_fn_40173?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block1_conv2_layer_call_and_return_conditional_losses_40184?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block1_pool_layer_call_fn_40189
+__inference_block1_pool_layer_call_fn_40194?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block1_pool_layer_call_and_return_conditional_losses_40199
F__inference_block1_pool_layer_call_and_return_conditional_losses_40204?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block2_conv1_layer_call_fn_40213?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block2_conv1_layer_call_and_return_conditional_losses_40224?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block2_conv2_layer_call_fn_40233?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block2_conv2_layer_call_and_return_conditional_losses_40244?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block2_pool_layer_call_fn_40249
+__inference_block2_pool_layer_call_fn_40254?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block2_pool_layer_call_and_return_conditional_losses_40259
F__inference_block2_pool_layer_call_and_return_conditional_losses_40264?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block3_conv1_layer_call_fn_40273?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block3_conv1_layer_call_and_return_conditional_losses_40284?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block3_conv2_layer_call_fn_40293?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block3_conv2_layer_call_and_return_conditional_losses_40304?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block3_conv3_layer_call_fn_40313?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block3_conv3_layer_call_and_return_conditional_losses_40324?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block3_conv4_layer_call_fn_40333?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block3_conv4_layer_call_and_return_conditional_losses_40344?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block3_pool_layer_call_fn_40349
+__inference_block3_pool_layer_call_fn_40354?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block3_pool_layer_call_and_return_conditional_losses_40359
F__inference_block3_pool_layer_call_and_return_conditional_losses_40364?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block4_conv1_layer_call_fn_40373?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block4_conv1_layer_call_and_return_conditional_losses_40384?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block4_conv2_layer_call_fn_40393?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block4_conv2_layer_call_and_return_conditional_losses_40404?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block4_conv3_layer_call_fn_40413?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block4_conv3_layer_call_and_return_conditional_losses_40424?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block4_conv4_layer_call_fn_40433?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block4_conv4_layer_call_and_return_conditional_losses_40444?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block4_pool_layer_call_fn_40449
+__inference_block4_pool_layer_call_fn_40454?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block4_pool_layer_call_and_return_conditional_losses_40459
F__inference_block4_pool_layer_call_and_return_conditional_losses_40464?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block5_conv1_layer_call_fn_40473?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block5_conv1_layer_call_and_return_conditional_losses_40484?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block5_conv2_layer_call_fn_40493?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block5_conv2_layer_call_and_return_conditional_losses_40504?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block5_conv3_layer_call_fn_40513?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block5_conv3_layer_call_and_return_conditional_losses_40524?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_block5_conv4_layer_call_fn_40533?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block5_conv4_layer_call_and_return_conditional_losses_40544?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block5_pool_layer_call_fn_40549
+__inference_block5_pool_layer_call_fn_40554?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block5_pool_layer_call_and_return_conditional_losses_40559
F__inference_block5_pool_layer_call_and_return_conditional_losses_40564?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_37521?"3456789:;<=>?@ABCDEFGHIJKLMNOPQR()>?;
4?1
/?,
vgg19_input???????????
? "-?*
(
dense?
dense??????????
G__inference_block1_conv1_layer_call_and_return_conditional_losses_40164p349?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????@
? ?
,__inference_block1_conv1_layer_call_fn_40153c349?6
/?,
*?'
inputs???????????
? ""????????????@?
G__inference_block1_conv2_layer_call_and_return_conditional_losses_40184p569?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0???????????@
? ?
,__inference_block1_conv2_layer_call_fn_40173c569?6
/?,
*?'
inputs???????????@
? ""????????????@?
F__inference_block1_pool_layer_call_and_return_conditional_losses_40199?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
F__inference_block1_pool_layer_call_and_return_conditional_losses_40204j9?6
/?,
*?'
inputs???????????@
? "-?*
#? 
0?????????@@@
? ?
+__inference_block1_pool_layer_call_fn_40189?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
+__inference_block1_pool_layer_call_fn_40194]9?6
/?,
*?'
inputs???????????@
? " ??????????@@@?
G__inference_block2_conv1_layer_call_and_return_conditional_losses_40224m787?4
-?*
(?%
inputs?????????@@@
? ".?+
$?!
0?????????@@?
? ?
,__inference_block2_conv1_layer_call_fn_40213`787?4
-?*
(?%
inputs?????????@@@
? "!??????????@@??
G__inference_block2_conv2_layer_call_and_return_conditional_losses_40244n9:8?5
.?+
)?&
inputs?????????@@?
? ".?+
$?!
0?????????@@?
? ?
,__inference_block2_conv2_layer_call_fn_40233a9:8?5
.?+
)?&
inputs?????????@@?
? "!??????????@@??
F__inference_block2_pool_layer_call_and_return_conditional_losses_40259?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
F__inference_block2_pool_layer_call_and_return_conditional_losses_40264j8?5
.?+
)?&
inputs?????????@@?
? ".?+
$?!
0?????????  ?
? ?
+__inference_block2_pool_layer_call_fn_40249?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
+__inference_block2_pool_layer_call_fn_40254]8?5
.?+
)?&
inputs?????????@@?
? "!??????????  ??
G__inference_block3_conv1_layer_call_and_return_conditional_losses_40284n;<8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0?????????  ?
? ?
,__inference_block3_conv1_layer_call_fn_40273a;<8?5
.?+
)?&
inputs?????????  ?
? "!??????????  ??
G__inference_block3_conv2_layer_call_and_return_conditional_losses_40304n=>8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0?????????  ?
? ?
,__inference_block3_conv2_layer_call_fn_40293a=>8?5
.?+
)?&
inputs?????????  ?
? "!??????????  ??
G__inference_block3_conv3_layer_call_and_return_conditional_losses_40324n?@8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0?????????  ?
? ?
,__inference_block3_conv3_layer_call_fn_40313a?@8?5
.?+
)?&
inputs?????????  ?
? "!??????????  ??
G__inference_block3_conv4_layer_call_and_return_conditional_losses_40344nAB8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0?????????  ?
? ?
,__inference_block3_conv4_layer_call_fn_40333aAB8?5
.?+
)?&
inputs?????????  ?
? "!??????????  ??
F__inference_block3_pool_layer_call_and_return_conditional_losses_40359?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
F__inference_block3_pool_layer_call_and_return_conditional_losses_40364j8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0??????????
? ?
+__inference_block3_pool_layer_call_fn_40349?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
+__inference_block3_pool_layer_call_fn_40354]8?5
.?+
)?&
inputs?????????  ?
? "!????????????
G__inference_block4_conv1_layer_call_and_return_conditional_losses_40384nCD8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block4_conv1_layer_call_fn_40373aCD8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_block4_conv2_layer_call_and_return_conditional_losses_40404nEF8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block4_conv2_layer_call_fn_40393aEF8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_block4_conv3_layer_call_and_return_conditional_losses_40424nGH8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block4_conv3_layer_call_fn_40413aGH8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_block4_conv4_layer_call_and_return_conditional_losses_40444nIJ8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block4_conv4_layer_call_fn_40433aIJ8?5
.?+
)?&
inputs??????????
? "!????????????
F__inference_block4_pool_layer_call_and_return_conditional_losses_40459?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
F__inference_block4_pool_layer_call_and_return_conditional_losses_40464j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_block4_pool_layer_call_fn_40449?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
+__inference_block4_pool_layer_call_fn_40454]8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_block5_conv1_layer_call_and_return_conditional_losses_40484nKL8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block5_conv1_layer_call_fn_40473aKL8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_block5_conv2_layer_call_and_return_conditional_losses_40504nMN8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block5_conv2_layer_call_fn_40493aMN8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_block5_conv3_layer_call_and_return_conditional_losses_40524nOP8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block5_conv3_layer_call_fn_40513aOP8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_block5_conv4_layer_call_and_return_conditional_losses_40544nQR8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block5_conv4_layer_call_fn_40533aQR8?5
.?+
)?&
inputs??????????
? "!????????????
F__inference_block5_pool_layer_call_and_return_conditional_losses_40559?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
F__inference_block5_pool_layer_call_and_return_conditional_losses_40564j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_block5_pool_layer_call_fn_40549?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
+__inference_block5_pool_layer_call_fn_40554]8?5
.?+
)?&
inputs??????????
? "!????????????
@__inference_dense_layer_call_and_return_conditional_losses_40144]()0?-
&?#
!?
inputs??????????@
? "%?"
?
0?????????
? y
%__inference_dense_layer_call_fn_40133P()0?-
&?#
!?
inputs??????????@
? "???????????
B__inference_flatten_layer_call_and_return_conditional_losses_40124b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????@
? ?
'__inference_flatten_layer_call_fn_40118U8?5
.?+
)?&
inputs??????????
? "???????????@?
E__inference_sequential_layer_call_and_return_conditional_losses_39171?"3456789:;<=>?@ABCDEFGHIJKLMNOPQR()F?C
<?9
/?,
vgg19_input???????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_39246?"3456789:;<=>?@ABCDEFGHIJKLMNOPQR()F?C
<?9
/?,
vgg19_input???????????
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_39603?"3456789:;<=>?@ABCDEFGHIJKLMNOPQR()A?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_39733?"3456789:;<=>?@ABCDEFGHIJKLMNOPQR()A?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
*__inference_sequential_layer_call_fn_38786?"3456789:;<=>?@ABCDEFGHIJKLMNOPQR()F?C
<?9
/?,
vgg19_input???????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_39096?"3456789:;<=>?@ABCDEFGHIJKLMNOPQR()F?C
<?9
/?,
vgg19_input???????????
p

 
? "???????????
*__inference_sequential_layer_call_fn_39400?"3456789:;<=>?@ABCDEFGHIJKLMNOPQR()A?>
7?4
*?'
inputs???????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_39473?"3456789:;<=>?@ABCDEFGHIJKLMNOPQR()A?>
7?4
*?'
inputs???????????
p

 
? "???????????
#__inference_signature_wrapper_39327?"3456789:;<=>?@ABCDEFGHIJKLMNOPQR()M?J
? 
C?@
>
vgg19_input/?,
vgg19_input???????????"-?*
(
dense?
dense??????????
@__inference_vgg19_layer_call_and_return_conditional_losses_38528? 3456789:;<=>?@ABCDEFGHIJKLMNOPQRB??
8?5
+?(
input_1???????????
p 

 
? ".?+
$?!
0??????????
? ?
@__inference_vgg19_layer_call_and_return_conditional_losses_38617? 3456789:;<=>?@ABCDEFGHIJKLMNOPQRB??
8?5
+?(
input_1???????????
p

 
? ".?+
$?!
0??????????
? ?
@__inference_vgg19_layer_call_and_return_conditional_losses_39992? 3456789:;<=>?@ABCDEFGHIJKLMNOPQRA?>
7?4
*?'
inputs???????????
p 

 
? ".?+
$?!
0??????????
? ?
@__inference_vgg19_layer_call_and_return_conditional_losses_40113? 3456789:;<=>?@ABCDEFGHIJKLMNOPQRA?>
7?4
*?'
inputs???????????
p

 
? ".?+
$?!
0??????????
? ?
%__inference_vgg19_layer_call_fn_37958? 3456789:;<=>?@ABCDEFGHIJKLMNOPQRB??
8?5
+?(
input_1???????????
p 

 
? "!????????????
%__inference_vgg19_layer_call_fn_38439? 3456789:;<=>?@ABCDEFGHIJKLMNOPQRB??
8?5
+?(
input_1???????????
p

 
? "!????????????
%__inference_vgg19_layer_call_fn_39802? 3456789:;<=>?@ABCDEFGHIJKLMNOPQRA?>
7?4
*?'
inputs???????????
p 

 
? "!????????????
%__inference_vgg19_layer_call_fn_39871? 3456789:;<=>?@ABCDEFGHIJKLMNOPQRA?>
7?4
*?'
inputs???????????
p

 
? "!???????????