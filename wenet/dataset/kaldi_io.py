#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2014-2016  Brno University of Technology (author: Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")

import numpy as np
import sys, os, re, gzip, struct

#################################################
# Adding kaldi tools to shell path,

# Select kaldi,
if not 'KALDI_ROOT' in os.environ:
    # Default! To change run python with 'export KALDI_ROOT=/some_dir python'
    os.environ['KALDI_ROOT'] = '/mnt/matylda5/iveselyk/Tools/kaldi-trunk'

# Add kaldi tools to path,
os.environ['PATH'] = os.popen(
    'echo $KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/lmbin/'
).readline().strip() + ':' + os.environ['PATH']


#################################################
# Define all custom exceptions,
class UnsupportedDataType(Exception):
    pass


class UnknownVectorHeader(Exception):
    pass


class UnknownMatrixHeader(Exception):
    pass


class BadSampleSize(Exception):
    pass


class BadInputFormat(Exception):
    pass


class SubprocessFailed(Exception):
    pass


#################################################
# Data-type independent helper functions,


# 其目的是灵活地打开文件、gzip压缩文件、管道，或者直接使用已打开的文件描述符（file descriptor，简称fd）
def open_or_fd(file, mode='rb'):
    """ fd = open_or_fd(file)
   Open file, gzipped file, pipe, or forward the file-descriptor.
   Eventually seeks in the 'file' argument contains ':offset' suffix.
  """
    offset = None
    try:
        # 使用正则表达式检查文件名是否以特定的前缀（如ark, scp等，后接可选的逗号分隔的修饰符）开头，但函数体内并未直接处理这些前缀。
        # strip 'ark:' prefix from r{x,w}filename (optional),
        if re.search('^(ark|scp)(,scp|,b|,t|,n?f|,n?p|,b?o|,n?s|,n?cs)*:',
                     file):
            (prefix, file) = file.split(':', 1)
        # 如果文件名中包含以冒号分隔的偏移量（位于文件名末尾且由数字组成），则将其与文件名分离。
        # separate offset from filename (optional),
        if re.search(':[0-9]+$', file):
            (file, offset) = file.rsplit(':', 1)
        # input pipe?
        # 如果文件名以竖线（|）结尾，则视为输入管道，使用 popen 函数（可能是自定义的，因为标准库中的 popen 已被弃用）以只读模式打开。
        if file[-1] == '|':
            fd = popen(file[:-1], 'rb')  # custom,
        # output pipe?
        # 如果文件名以竖线（|）开头，则视为输出管道，使用 popen 函数以只写模式打开。
        elif file[0] == '|':
            fd = popen(file[1:], 'wb')  # custom,
        # is it gzipped?
        # 如果文件名以 .gz 结尾，则使用 gzip.open 函数以指定的模式打开gzip压缩文件。
        elif file.split('.')[-1] == 'gz':
            fd = gzip.open(file, mode)
        # a normal file...
        # 如果文件名不符合上述任何特殊情况，则使用内置的 open 函数以指定的模式打开文件。只是一个普通的文件
        else:
            fd = open(file, mode)
    # 如果在尝试打开文件时引发 TypeError 异常（这通常发生在 file 参数已是一个打开的文件描述符而非字符串时），则直接返回该文件描述符。
    except TypeError:
        # 'file' is opened file descriptor,
        fd = file
    # Eventually seek to offset,
    # 如果在文件名中识别到偏移量，则使用 seek 方法将文件指针移动到指定的偏移位置。
    if offset != None: fd.seek(int(offset))
    # 函数返回打开的文件对象（或文件描述符），如果文件名中包含偏移量，则文件指针已被移动到该偏移位置。
    return fd


# 模拟 Unix/Linux 系统中的 popen 函数行为（进程间通信）
# based on '/usr/local/lib/python3.4/os.py'
def popen(cmd, mode="rb"):
    # 首先检查 cmd 参数是否为字符串类型。如果不是，将引发 TypeError 异常。
    if not isinstance(cmd, str):
        raise TypeError("invalid cmd type (%s, expected string)" % type(cmd))

    # 函数内部导入了 subprocess、io 和 threading 模块，这些模块分别用于创建子进程、处理文件流和线程管理。
    import subprocess, io, threading

    # cleanup function for subprocesses,
    # cleanup 函数是一个内部定义的函数，用于等待子进程结束并检查其退出状态。
    # 如果子进程返回非零值（通常表示错误），则引发 SubprocessFailed 异常（注意：这个异常类在代码片段中未定义，需要在外部定义或替换为其他异常处理机制）。
    def cleanup(proc, cmd):
        ret = proc.wait()
        if ret > 0:
            raise SubprocessFailed('cmd %s returned %d !' % (cmd, ret))
        return

    # text-mode,
    # 如果 mode 为 "r" 或 "rb"，函数将创建一个子进程来执行 cmd 命令，并将 stdout 重定向到管道。
    # 对于 "r" 模式，返回的是一个文本模式的文件对象（通过 io.TextIOWrapper 包装 proc.stdout）
    if mode == "r":
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        threading.Thread(target=cleanup,
                         args=(proc, cmd)).start()  # clean-up thread,
        return io.TextIOWrapper(proc.stdout)
    # 如果 mode 为 "w" 或 "wb"，函数同样创建一个子进程，但这次是将 stdin 重定向到管道。
    # 对于 "w" 模式，返回的是一个文本模式的文件对象
    elif mode == "w":
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE)
        threading.Thread(target=cleanup,
                         args=(proc, cmd)).start()  # clean-up thread,
        return io.TextIOWrapper(proc.stdin)
    # binary,
    # 对于 "rb" 模式，直接返回二进制模式的文件对象（proc.stdout）。
    elif mode == "rb":
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        threading.Thread(target=cleanup,
                         args=(proc, cmd)).start()  # clean-up thread,
        return proc.stdout
    # 对于 "wb" 模式，直接返回二进制模式的文件对象（proc.stdin）。
    elif mode == "wb":
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE)
        threading.Thread(target=cleanup,
                         args=(proc, cmd)).start()  # clean-up thread,
        return proc.stdin
    # sanity,否则返回错误
    else:
        raise ValueError("invalid mode %s" % mode)


# 用于从打开的文件描述符 fd 中读取一个键值（utterance-key）。
# 该函数通过逐字符读取文件内容，直到遇到空字符或空格为止，然后将读取的字符拼接成一个字符串，并去除字符串两端的空白字符。
# 如果读取的字符串为空，则返回 None，表示文件结束。
# 最后，使用正则表达式 re.match('^\S+$', key) 检查字符串是否只包含非空白字符，以确保格式正确。
def read_key(fd):
    """ [key] = read_key(fd)
   Read the utterance-key from the opened ark/stream descriptor 'fd'.
  """
    key = ''
    while 1:
        # 从文件描述符 fd 中读取一个字符，并将其解码为字符串。latin1 编码用于处理非ASCII字符。
        char = fd.read(1).decode("latin1")
        # 如果读取到空字符或空格，则停止读取，表示已经读取到键值的末尾。
        if char == '': break
        if char == ' ': break
        # 将读取到的字符拼接到 key 字符串中。
        key += char
    # 使用 strip() 方法去除字符串两端的空白字符。
    key = key.strip()
    # 使用正则表达式 ^\S+$ 检查 key 是否只包含非空白字符。^\S+$ 表示字符串必须完全由非空白字符组成。
    if key == '': return None  # end of file,
    assert (re.match('^\S+$', key) != None)  # check format (no whitespace!)
    # 如果格式正确，则返回 key。
    return key


#################################################
# Integer vectors (alignments, ...),


# read_vec_int_ark()的别名
def read_ali_ark(file_or_fd):
    """ Alias to 'read_vec_int_ark()' """
    return read_vec_int_ark(file_or_fd)


# 生成器（Generator）是一种特殊的迭代器，它允许在需要时逐个生成值，而不是一次性生成所有值。
def read_vec_int_ark(file_or_fd):
    """ generator(key,vec) = read_vec_int_ark(file_or_fd)
   Create generator of (key,vector<int>) tuples, which reads from the ark file/stream.
   file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

   Read ark to a 'dictionary':
   d = { u:d for u,d in kaldi_io.read_vec_int_ark(file) }
  """
    # 它从一个ark文件或流中读取数据，并逐个生成键值对（key, vector<int>)。
    # 首先打开文件或文件描述符
    fd = open_or_fd(file_or_fd)
    try:
        # 然后通过循环读取每个键和对应的向量
        key = read_key(fd)
        while key:
            # 每次读取后使用yield关键字返回一个键值对
            ali = read_vec_int(fd)
            yield key, ali
            key = read_key(fd)
    # 最后，无论是否成功读取完所有数据，都会确保文件被正确关闭。
    finally:
        if fd is not file_or_fd: fd.close()


# 从一个符合 Kaldi scp 格式的文件中读取数据，并生成一个包含键值对的生成器。
# 这个函数支持直接从文件、gzip压缩的文件、管道或者已经打开的文件描述符中读取数据。
# 函数的参数 file_or_fd 可以是一个文件路径、gzip压缩的文件路径、一个管道对象或者是一个已经打开的文件描述符。
def read_vec_int_scp(file_or_fd):
    """ generator(key,vec) = read_vec_int_scp(file_or_fd)
   Returns generator of (key,vector<int>) tuples, read according to kaldi scp.
   file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

   Iterate the scp:
   for key,vec in kaldi_io.read_vec_int_scp(file):
     ...

   Read scp to a 'dictionary':
   d = { key:vec for key,mat in kaldi_io.read_vec_int_scp(file) }
  """
    # 使用 open_or_fd 函数（该函数未在代码中定义，但基于上下文，我们可以推断它用于打开文件或者接受一个已经打开的文件描述符）打开 file_or_fd 指定的文件或文件描述符，并将其赋值给变量 fd。
    fd = open_or_fd(file_or_fd)
    # 使用 try 语句块来确保文件或文件描述符在操作完成后能够正确关闭。
    try:
        # 逐行读取 fd 中的内容。每一行应该包含一个键值对，其中键和值（一个指向实际数据文件的路径）之间用空格分隔。
        for line in fd:
            # 对每一行，首先使用 decode() 方法将行内容解码为字符串，然后使用 split(' ') 方法将行分割成键和值两部分。
            (key, rxfile) = line.decode().split(' ')
            # 调用 read_vec_int 函数，读取整数向量，其中路径是步骤4中得到的值。
            vec = read_vec_int(rxfile)
            # 使用 yield 关键字生成一个包含键和整数向量的元组。
            yield key, vec
    finally:
        # 在 finally 语句块中，检查如果 fd 不是原始传入的 file_or_fd（即如果 fd 是函数内部打开的），则关闭它。
        # 这是为了确保无论函数是否因异常而提前退出，文件都能被正确关闭。
        if fd is not file_or_fd: fd.close()


# 用于读取 Kaldi 格式的整数向量，这些向量可以以 ASCII 或二进制格式存储。
# 函数接受一个参数 file_or_fd，它可以是文件路径、gzip 压缩的文件路径、管道对象或已打开的文件描述符。
def read_vec_int(file_or_fd):
    """ [int-vec] = read_vec_int(file_or_fd)
   Read kaldi integer vector, ascii or binary input,
  """
    # 使用 open_or_fd 函数（未在代码中定义，但预期用于打开文件或接受文件描述符）打开 file_or_fd 指定的文件或文件描述符，并将其赋值给变量 fd。
    fd = open_or_fd(file_or_fd)
    # 读取二进制标志：从 fd 中读取前两个字节，并尝试解码为字符串。
    binary = fd.read(2).decode()
    # 如果前两个字节解码后是 '\0B'，则表明接下来的数据是二进制格式的。
    if binary == '\0B':  # binary flag
        # 验证接下来的一个字节是否为 '\4'（这可能是用于进一步确认二进制格式的标志）。
        assert (fd.read(1).decode() == '\4')
        # int-size
        # 从 fd 中读取接下来的 4 个字节，并将其解释为一个 32 位整数（int32），这个整数表示向量的维度（即向量中元素的数量）。
        vec_size = np.frombuffer(fd.read(4), dtype='int32',
                                 count=1)[0]  # vector dim
        # Elements from int32 vector are sored in tuples: (sizeof(int32), value),
        # 根据向量的维度，从 fd 中读取相应数量的数据。每个数据项都是一个由 1 个字节的大小信息和 4 个字节的整数值组成的元组。
        # 这里的大小信息固定为 4（表示 int32 的大小），而整数值是实际的向量元素。
        vec = np.frombuffer(fd.read(vec_size * 5),
                            dtype=[('size', 'int8'), ('value', 'int32')],
                            count=vec_size)
        # 验证第一个元素的大小信息是否为 4，以确保数据格式正确。
        assert (vec[0]['size'] == 4)  # int32 size,
        # 从读取的数据中提取整数值，并存储在数组 ans 中。
        ans = vec[:]['value']  # values are in 2nd column,
    else:  # ascii,
        # 如果前两个字节不是 '\0B'，则假定数据是 ASCII 格式的。
        arr = (binary + fd.readline().decode()).strip().split()
        try:
            # 读取一行数据，并尝试去除可能的 [ 和 ] 字符（这些字符在 ASCII 格式的向量表示中可能用作包围向量的括号）。
            arr.remove('[')
            arr.remove(']')  # optionally
        except ValueError:
            pass
        # 将剩余的数据分割成字符串列表，并将这些字符串转换为整数数组 ans。
        ans = np.array(arr, dtype=int)
    # 如果 fd 不是原始传入的 file_or_fd（即如果 fd 是函数内部打开的），则关闭它。
    if fd is not file_or_fd: fd.close()  # cleanup
    # 返回包含整数向量的数组 ans。
    return ans


# Writing,
# 这段代码定义了一个名为 write_vec_int 的函数，用于将整数向量以 Kaldi 的二进制格式写入文件或文件描述符。
def write_vec_int(file_or_fd, v, key=''):
    """ write_vec_int(f, v, key='')
   Write a binary kaldi integer vector to filename or stream.
   Arguments:
   file_or_fd : filename or opened file descriptor for writing,
   v : the vector to be stored,
   key (optional) : used for writing ark-file, the utterance-id gets written before the vector.

   Example of writing single vector:
   kaldi_io.write_vec_int(filename, vec)

   Example of writing arkfile:
   with open(ark_file,'w') as f:
     for key,vec in dict.iteritems():
       kaldi_io.write_vec_flt(f, vec, key=key)
  """
  # 使用 open_or_fd 函数（未在代码中定义，但预期用于打开文件或接受文件描述符，并设置写入模式为 'wb'）打开 file_or_fd 指定的文件或文件描述符，并将其赋值给变量 fd。
    fd = open_or_fd(file_or_fd, mode='wb')
    # 检查 Python 版本，确保在 Python 3 中文件是以二进制写入模式打开的（'wb'）。
    if sys.version_info[0] == 3: assert (fd.mode == 'wb')
    # 如果提供了 key（不为空字符串），则将其与空格一起编码为 Latin-1 格式，并写入文件。
    # 这是为了在处理 ark 文件时写入话语 ID（utterance-id）。如果提供了 key（不为空字符串），则将其与空格一起编码为 Latin-1 格式，并写入文件。这是为了在处理 ark 文件时写入话语 ID（utterance-id）。
    try:
        if key != '':
            fd.write(
                (key +
                 ' ').encode("latin1"))  # ark-files have keys (utterance-id),
        # 向文件中写入两个字节的二进制标志 '\0B'（编码为字节）。
        fd.write('\0B'.encode())  # we write binary!
        # dim,
        # dim,写入一个字节的整数类型标志 '\4'（表示 int32 类型，编码为字节）。
        fd.write('\4'.encode())  # int32 type,
        # 使用 struct.pack 函数和 NumPy 的数据类型字符将向量的维度（v.shape[0]）打包为 int32 类型的二进制数据，并写入文件。
        fd.write(struct.pack(np.dtype('int32').char, v.shape[0]))
        # data,
        # 使用 struct.pack 函数和 NumPy 的数据类型字符将该元素打包为 int32 类型的二进制数据，并写入文件。
        for i in range(len(v)):
            fd.write('\4'.encode())  # int32 type,
            fd.write(struct.pack(np.dtype('int32').char, v[i]))  # binary,
    # 如果 fd 不是原始传入的 file_or_fd（即如果 fd 是函数内部打开的），则关闭它。
    finally:
        if fd is not file_or_fd: fd.close()


#################################################
# Float vectors (confidences, ivectors, ...),


# Reading,
# 读取 Kaldi scp 文件中的向量数据，并返回一个生成器，该生成器生成 (key, vector) 元组。
def read_vec_flt_scp(file_or_fd):
    """ generator(key,mat) = read_vec_flt_scp(file_or_fd)
   Returns generator of (key,vector) tuples, read according to kaldi scp.
   file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

   Iterate the scp:
   for key,vec in kaldi_io.read_vec_flt_scp(file):
     ...

   Read scp to a 'dictionary':
   d = { key:mat for key,mat in kaldi_io.read_mat_scp(file) }
  """
    #  打开文件或文件描述符，并将其存储在 fd 变量中。
    fd = open_or_fd(file_or_fd)
    # 确保在任何情况下都能正确关闭文件描述符。
    try:
        # 使用 for line in fd 逐行读取文件内容。
        for line in fd:
            # 每行内容通过 decode().split(' ') 解析为键和向量文件路径。
            (key, rxfile) = line.decode().split(' ')
            # 使用 read_vec_flt(rxfile) 读取向量数据，并将其与键一起返回。
            vec = read_vec_flt(rxfile)
            # 返回结果
            yield key, vec
    finally:
        if fd is not file_or_fd: fd.close()


# 用于从给定的文件或文件描述符（可能是一个ARK文件、gzip压缩的ARK文件、管道或已打开的文件描述符）中读取键值对，
def read_vec_flt_ark(file_or_fd):
    """ generator(key,vec) = read_vec_flt_ark(file_or_fd)
   Create generator of (key,vector<float>) tuples, reading from an ark file/stream.
   file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

   Read ark to a 'dictionary':
   d = { u:d for u,d in kaldi_io.read_vec_flt_ark(file) }
  """
  # 打开文件或文件描述符，并将其存储在 fd 变量中。
    fd = open_or_fd(file_or_fd)
    # 使用 try 语句来确保文件最终会被正确关闭。
    try:
        # 首先调用 read_key 函数（同样未在代码中定义，但可以推测它的作用是从文件中读取一个键）来读取第一个键，并将其存储在变量 key 中。
        key = read_key(fd)
        # 进入一个循环，只要 key 不为空（即文件还未结束）
        while key:
            # 从文件中读取一个浮点数向量
            ali = read_vec_flt(fd)
            # 使用 yield 关键字生成一个（key, ali）元组，这样调用者可以逐个获取这些元组。
            yield key, ali
            # 再次调用 read_key 函数读取下一个键，为下一次循环做准备。
            key = read_key(fd)
    finally:
        # 如果 fd 不是原始的 file_or_fd（即如果 open_or_fd 函数打开了一个新的文件描述符），则关闭 fd。
        if fd is not file_or_fd: fd.close()


# 用于读取 Kaldi 格式的浮点向量
def read_vec_flt(file_or_fd):
    """ [flt-vec] = read_vec_flt(file_or_fd)
   Read kaldi float vector, ascii or binary input,
  """
    # 该函数接受一个文件对象或文件路径作为输入，并返回一个浮点数向量。
    fd = open_or_fd(file_or_fd)
    # 使用 open_or_fd 函数打开文件或文件对象。
    binary = fd.read(2).decode()
    # 读取前两个字节并解码为字符串，判断是否为二进制格式。
    if binary == '\0B':  # binary flag
        # Data type,读取接下来的三个字节以确定数据类型（浮点数或双精度数）。
        header = fd.read(3).decode()
        if header == 'FV ': sample_size = 4  # floats
        elif header == 'DV ': sample_size = 8  # doubles
        else: raise UnknownVectorHeader("The header contained '%s'" % header)
        assert (sample_size > 0)
        # Dimension,读取一个字节并断言其值为 \4，然后读取接下来的四个字节以获取向量的维度。
        assert (fd.read(1).decode() == '\4')
        # int-size
        vec_size = np.frombuffer(fd.read(4), dtype='int32',
                                 count=1)[0]  # vector dim
        # Read whole vector,根据数据类型读取整个向量，并将其转换为 NumPy 数组。
        buf = fd.read(vec_size * sample_size)
        # 根据不同的长度转换为对应的数据格式
        if sample_size == 4: ans = np.frombuffer(buf, dtype='float32')
        elif sample_size == 8: ans = np.frombuffer(buf, dtype='float64')
        # 如果都无法匹配就算样例错误
        else: raise BadSampleSize
        # 返回浮点向量结果
        return ans
    else:  # ascii,如果不是二进制格式，则读取一行 ASCII 数据，去除方括号（可选），并将其转换为 NumPy 数组。
        arr = (binary + fd.readline().decode()).strip().split()
        try:
            arr.remove('[')
            arr.remove(']')  # optionally
        except ValueError:
            pass
        ans = np.array(arr, dtype=float)
    if fd is not file_or_fd: fd.close()  # cleanup,清除数据保证安全
    return ans


# Writing,将 Kaldi 格式的浮点数向量（支持32位和64位浮点数）写入文件或文件描述符的。
def write_vec_flt(file_or_fd, v, key=''):
    """ write_vec_flt(f, v, key='')
   Write a binary kaldi vector to filename or stream. Supports 32bit and 64bit floats.
   Arguments:
   file_or_fd : filename or opened file descriptor for writing,
   v : the vector to be stored,
   key (optional) : used for writing ark-file, the utterance-id gets written before the vector.

   Example of writing single vector:
   kaldi_io.write_vec_flt(filename, vec)

   Example of writing arkfile:
   with open(ark_file,'w') as f:
     for key,vec in dict.iteritems():
       kaldi_io.write_vec_flt(f, vec, key=key)
  """
    # 函数打开或获取文件名称（在file_or_fd里)，二进制写入
    fd = open_or_fd(file_or_fd, mode='wb')
    # 如果使用的是Python 3，则确保文件模式为 'wb'
    if sys.version_info[0] == 3: assert (fd.mode == 'wb')
    try:
        if key != '':
            fd.write(
                (key +
                 ' ').encode("latin1"))  # ark-files have keys (utterance-id),,如果提供了 key 参数，则将其编码为latin1格式并写入文件。这通常用于ark文件，其中utterance-id会在向量之前写入。
        # 写入二进制标识
        fd.write('\0B'.encode())  # we write binary!
        # Data-type,根据向量 v 的数据类型（dtype），函数会写入相应的标识符：
        # 如果 v.dtype 是 'float32'，则写入 'FV '。
        if v.dtype == 'float32': fd.write('FV '.encode())
        # 如果 v.dtype 是 'float64'，则写入 'DV '。
        elif v.dtype == 'float64': fd.write('DV '.encode())
        # 都不是抛出 UnsupportedDataType 异常。
        else:
            raise UnsupportedDataType(
                "'%s', please use 'float32' or 'float64'" % v.dtype)
        # Dim,将向量的维度（v.shape[0]）打包为二进制数据并写入文件
        fd.write('\04'.encode())
        fd.write(struct.pack(np.dtype('uint32').char, v.shape[0]))  # dim
        # Data,函数将向量 v 的二进制数据写入文件。
        fd.write(v.tobytes())
    finally:
        # 关闭文件写入
        if fd is not file_or_fd: fd.close()


#################################################
# Float matrices (features, transformations, ...),


# Reading,从kaldi scp读取文件
def read_mat_scp(file_or_fd):
    """ generator(key,mat) = read_mat_scp(file_or_fd)
   Returns generator of (key,matrix) tuples, read according to kaldi scp.
   file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

   Iterate the scp:
   for key,mat in kaldi_io.read_mat_scp(file):
     ...

   Read scp to a 'dictionary':
   d = { key:mat for key,mat in kaldi_io.read_mat_scp(file) }
  """
    # 打开文件或文件描述符。
    fd = open_or_fd(file_or_fd)
    try:
        # 按行读取，用空格分开关键字和值
        for line in fd:
            (key, rxfile) = line.decode().split(' ')
            # 调用read_mat函数处理值部分，并生成一个包含键和处理后值的元组。
            mat = read_mat(rxfile)
            # 使用 yield 关键字生成一个包含键和矩阵的元组
            # 在迭代过程中逐步生成值，而不是一次性生成所有值
            yield key, mat
    # 最后关闭读取
    finally:
        if fd is not file_or_fd: fd.close()


# 从一个 Kaldi ark 文件中读取数据，生成关键字和矩阵对应的表
# ark 文件是 Kaldi 语音识别工具包中用于存储序列化对象（如矩阵）的一种二进制格式
# 与 scp 文件不同，ark 文件不存储文件路径，而是直接存储序列化后的数据。
def read_mat_ark(file_or_fd):
    """ generator(key,mat) = read_mat_ark(file_or_fd)
   Returns generator of (key,matrix) tuples, read from ark file/stream.
   file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

   Iterate the ark:
   for key,mat in kaldi_io.read_mat_ark(file):
     ...

   Read ark to a 'dictionary':
   d = { key:mat for key,mat in kaldi_io.read_mat_ark(file) }
  """
  # 打开文件
    fd = open_or_fd(file_or_fd)
    try:
        # 读取关键字
        key = read_key(fd)
        while key:
            mat = read_mat(fd)
            yield key, mat
            key = read_key(fd)
    # 检查 fd 是否不是原始传入的 file_or_fd（即 fd 是通过 open_or_fd 打开的），如果是，则关闭 fd。这是为了确保即使发生异常，文件也能被正确关闭。
    finally:
        if fd is not file_or_fd: fd.close()


# 用于读取 Kaldi 矩阵文件，支持 ASCII 和二进制格式。
def read_mat(file_or_fd):
    """ [mat] = read_mat(file_or_fd)
   Reads single kaldi matrix, supports ascii and binary.
   file_or_fd : file, gzipped file, pipe or opened file descriptor.
  """
    # 打开文件
    fd = open_or_fd(file_or_fd)
    # 使用try确保能关闭
    try:
        # 读取前三位并解码
        binary = fd.read(2).decode()
        # 开头是0B的话直接读取成二进制格式
        if binary == '\0B':
            mat = _read_mat_binary(fd)
        # 否则先确定有没有[，再按照ascii格式读取
        else:
            assert (binary == ' [')
            mat = _read_mat_ascii(fd)
    # 检查 fd 是否不是原始传入的 file_or_fd（即 fd 是通过 open_or_fd 打开的），如果是，则关闭 fd。这是为了确保即使发生异常，文件也能被正确关闭。
    finally:
        if fd is not file_or_fd: fd.close()
    # 返回矩阵
    return mat


# 读取kaild矩阵文件，二进制形式的kaldi文件
def _read_mat_binary(fd):
    # Data type
    header = fd.read(3).decode()
    # 'CM', 'CM2', 'CM3' are possible values,
    # Data type，从文件中读取3个字节，并将其解码为字符串，查看文件格式
    if header.startswith('CM'): return _read_compressed_mat(fd, header)
    # FM开头的话，表示数据类型为浮点数（4字节）
    elif header == 'FM ': sample_size = 4  # floats
    # DM开头的话，则表示数据类型为双精度浮点数（8字节）
    elif header == 'DM ': sample_size = 8  # doubles
    # 以上都不是的话抛出异常无法识别
    else: raise UnknownMatrixHeader("The header contained '%s'" % header)
    # assert确保例子存在
    assert (sample_size > 0)
    # Dimensions从文件中读取10个字节，并将其解释为一个包含4个元素的数组。
    # dtype='int8,int32,int8,int32' 指定了每个元素的数据类型，分别是 int8, int32, int8, int32。
    # count=1 表示只读取一次。
    s1, rows, s2, cols = np.frombuffer(fd.read(10),
                                       dtype='int8,int32,int8,int32',
                                       count=1)[0]
    # Read whole matrix
    # 根据之前确定的 rows 和 cols 以及 sample_size，计算出矩阵数据的总大小，并从文件中读取这些数据。
    buf = fd.read(rows * cols * sample_size)
    # 根据 sample_size 的值，将读取的二进制数据转换为NumPy数组。如果 sample_size 是4，则数据类型为 float32；
    # 如果 sample_size 是8，则数据类型为 float64。如果 sample_size 不是4或8，则抛出 BadSampleSize 异常。
    if sample_size == 4: vec = np.frombuffer(buf, dtype='float32')
    elif sample_size == 8: vec = np.frombuffer(buf, dtype='float64')
    else: raise BadSampleSize
    # 使用 np.reshape 函数将一维数组 vec 重塑为二维数组 mat，形状为 (rows, cols)。np.reshape 函数的作用是调整数组的形状，而不改变其数据内容。
    mat = np.reshape(vec, (rows, cols))
    # 返回重塑后的矩阵
    return mat


# 按照ascii格式读取矩阵
def _read_mat_ascii(fd):
    rows = []
    while 1:
        line = fd.readline().decode()
        if (len(line) == 0): raise BadInputFormat  # eof, should not happen!空的不应该存在
        if len(line.strip()) == 0: continue  # skip empty line跳过空行
        # 数组是通过空行空格分开的，读取每行的元素记入数组
        arr = line.strip().split()
        # 如果不是最后一行，则将 arr（转换为 float32 类型的NumPy数组）添加到 rows 列表中。
        if arr[-1] != ']':
            rows.append(np.array(arr, dtype='float32'))  # not last line
        # 是最后一行的话，则将 arr 中除了最后一个元素（']'）之外的所有元素转换为 float32 类型的NumPy数组，并添加到 rows 列表中。
        else:
            rows.append(np.array(arr[:-1], dtype='float32'))  # last line
            # 当遇到最后一行时，使用 np.vstack(rows) 将 rows 列表中的所有数组垂直堆叠成一个矩阵，然后返回这个矩阵。
            mat = np.vstack(rows)
            return mat


# 读取压缩格式的矩阵数据，这种压缩格式由kaldi定义
def _read_compressed_mat(fd, format):
    """ Read a compressed matrix,
      see: https://github.com/kaldi-asr/kaldi/blob/master/src/matrix/compressed-matrix.h
      methods: CompressedMatrix::Read(...), CompressedMatrix::CopyToMat(...),
  """
    assert (format == 'CM ')  # The formats CM2, CM3 are not supported...

    # Format of header 'struct',
    global_header = np.dtype([('minvalue', 'float32'), ('range', 'float32'),
                              ('num_rows', 'int32'), ('num_cols', 'int32')
                              ])  # member '.format' is not written,
    per_col_header = np.dtype([('percentile_0', 'uint16'),
                               ('percentile_25', 'uint16'),
                               ('percentile_75', 'uint16'),
                               ('percentile_100', 'uint16')])

    # Mapping for percentiles in col-headers,unit16无符号整数转float
    def uint16_to_float(value, min, range):
        return np.float32(min + range * 1.52590218966964e-05 * value)

    # Mapping for matrix elements,unit8无符号整数转float第二版
    def uint8_to_float_v2(vec, p0, p25, p75, p100):
        # Split the vector by masks,
        mask_0_64 = (vec <= 64)
        mask_193_255 = (vec > 192)
        mask_65_192 = (~(mask_0_64 | mask_193_255))
        # Sanity check (useful but slow...),
        # assert(len(vec) == np.sum(np.hstack([mask_0_64,mask_65_192,mask_193_255])))
        # assert(len(vec) == np.sum(np.any([mask_0_64,mask_65_192,mask_193_255], axis=0)))
        # Build the float vector,
        ans = np.empty(len(vec), dtype='float32')
        ans[mask_0_64] = p0 + (p25 - p0) / 64. * vec[mask_0_64]
        ans[mask_65_192] = p25 + (p75 - p25) / 128. * (vec[mask_65_192] - 64)
        ans[mask_193_255] = p75 + (p100 - p75) / 63. * (vec[mask_193_255] -
                                                        192)
        return ans

    # Read global header,从文件描述符 fd 中读取16字节的数据，并将其解析为 global_header 类型的数据。
    globmin, globrange, rows, cols = np.frombuffer(fd.read(16),
                                                   dtype=global_header,
                                                   count=1)[0]

    # The data is structed as [Colheader, ... , Colheader, Data, Data , .... ]
    #                         {           cols           }{     size         }
    # 根据全局头部中的列数（cols），从文件中读取相应数量的字节，并使用 per_col_header 数据结构解析为每列头部的各个字段。
    col_headers = np.frombuffer(fd.read(cols * 8),
                                dtype=per_col_header,
                                count=cols)
    data = np.reshape(np.frombuffer(fd.read(cols * rows),
                                    dtype='uint8',
                                    count=cols * rows),
                      newshape=(cols, rows))  # stored as col-major,

    mat = np.empty((cols, rows), dtype='float32')
    for i, col_header in enumerate(col_headers):
        col_header_flt = [
            uint16_to_float(percentile, globmin, globrange)
            for percentile in col_header
        ]
        mat[i] = uint8_to_float_v2(data[i], *col_header_flt)

    return mat.T  # transpose! col-major -> row-major,


# Writing,将矩阵以 Kaldi 的二进制格式写入文件或文件描述符的函数。创建 ark 文件
def write_ark_scp(key, mat, ark_fout, scp_out):
    mat_offset = write_mat(ark_fout, mat, key)
    scp_line = '{}\t{}:{}'.format(key, ark_fout.name, mat_offset)
    scp_out.write(scp_line)
    scp_out.write('\n')


# Writing,
def write_mat(file_or_fd, m, key=''):
    """ write_mat(f, m, key='')
  Write a binary kaldi matrix to filename or stream. Supports 32bit and 64bit floats.
  Arguments:
   file_or_fd : filename of opened file descriptor for writing,
   m : the matrix to be stored,
   key (optional) : used for writing ark-file, the utterance-id gets written before the matrix.

   Example of writing single matrix:
   kaldi_io.write_mat(filename, mat)

   Example of writing arkfile:
   with open(ark_file,'w') as f:
     for key,mat in dict.iteritems():
       kaldi_io.write_mat(f, mat, key=key)
  """
    mat_offset = 0
    fd = open_or_fd(file_or_fd, mode='wb')
    if sys.version_info[0] == 3: assert (fd.mode == 'wb')
    try:
        if key != '':
            fd.write(
                (key +
                 ' ').encode("latin1"))  # ark-files have keys (utterance-id),
        mat_offset = fd.tell()
        fd.write('\0B'.encode())  # we write binary!
        # Data-type,
        if m.dtype == 'float32': fd.write('FM '.encode())
        elif m.dtype == 'float64': fd.write('DM '.encode())
        else:
            raise UnsupportedDataType(
                "'%s', please use 'float32' or 'float64'" % m.dtype)
        # Dims,
        fd.write('\04'.encode())
        fd.write(struct.pack(np.dtype('uint32').char, m.shape[0]))  # rows
        fd.write('\04'.encode())
        fd.write(struct.pack(np.dtype('uint32').char, m.shape[1]))  # cols
        # Data,
        fd.write(m.tobytes())
    finally:
        if fd is not file_or_fd: fd.close()
    return mat_offset


#################################################
# 'Posterior' kaldi type (posteriors, confusion network, nnet1 training targets, ...)
# Corresponds to: vector<vector<tuple<int,float> > >
# - outer vector: time axis
# - inner vector: records at the time
# - tuple: int = index, float = value
#


# 读取混淆网络
def read_cnet_ark(file_or_fd):
    """ Alias of function 'read_post_ark()', 'cnet' = confusion network """
    return read_post_ark(file_or_fd)


# 读取 后验概率ark文件
def read_post_ark(file_or_fd):
    """ generator(key,vec<vec<int,float>>) = read_post_ark(file)
   Returns generator of (key,posterior) tuples, read from ark file.
   file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

   Iterate the ark:
   for key,post in kaldi_io.read_post_ark(file):
     ...

   Read ark to a 'dictionary':
   d = { key:post for key,post in kaldi_io.read_post_ark(file) }
  """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            post = read_post(fd)
            yield key, post
            key = read_key(fd)
    finally:
        if fd is not file_or_fd: fd.close()


# 从 Kaldi 格式的二进制文件中读取单个“Posterior”（后验概率）数据
def read_post(file_or_fd):
    """ [post] = read_post(file_or_fd)
   Reads single kaldi 'Posterior' in binary format.

   The 'Posterior' is C++ type 'vector<vector<tuple<int,float> > >',
   the outer-vector is usually time axis, inner-vector are the records
   at given time,  and the tuple is composed of an 'index' (integer)
   and a 'float-value'. The 'float-value' can represent a probability
   or any other numeric value.

   Returns vector of vectors of tuples.
  """
    fd = open_or_fd(file_or_fd)
    ans = []
    # 读取前三个字节
    binary = fd.read(2).decode()
    # 断言是\0B，确保文件头符合 Kaldi 的二进制格式标志。
    assert (binary == '\0B')
    # binary flag读取一个字节并验证其是否为整数大小的标志（'\4' 通常表示 4 字节整数）。
    assert (fd.read(1).decode() == '\4')
    # int-size，使用 np.frombuffer 从文件中读取 4 个字节，并将其解释为 32 位整数（dtype='int32'），表示外层向量（通常对应时间轴）的大小。
    outer_vec_size = np.frombuffer(fd.read(4), dtype='int32',
                                   count=1)[0]  # number of frames (or bins)

    # Loop over 'outer-vector',遍历外层向量
    for i in range(outer_vec_size):
        # 验证整数大小标志
        assert (fd.read(1).decode() == '\4')
        # int-size读取并解析内层向量的大小
        inner_vec_size = np.frombuffer(
            fd.read(4), dtype='int32',
            count=1)[0]  # number of records for frame (or bin)
        # 使用 np.frombuffer 读取并解析这些记录
        data = np.frombuffer(fd.read(inner_vec_size * 10),
                             dtype=[('size_idx', 'int8'), ('idx', 'int32'),# 索引大小（固定为 4 字节）# 索引大小（固定为 4 字节）
                                    ('size_post', 'int8'),# 后验概率大小（固定为 4 字节）
                                    ('post', 'float32')], # 后验概率值（32 位浮点数）
                             count=inner_vec_size)
        # 验证 size_idx 和 size_post 是否符合预期（均为 4）
        assert (data[0]['size_idx'] == 4)
        assert (data[0]['size_post'] == 4)
        # 将 idx 和 post 添加到答案列表 ans 中。
        ans.append(data[['idx', 'post']].tolist())
    # 如果 fd 不是原始传入的 file_or_fd（即如果 open_or_fd 打开了一个新文件描述符），则关闭它。
    if fd is not file_or_fd: fd.close()
    return ans


#################################################
# Kaldi Confusion Network bin begin/end times,
# (kaldi stores CNs time info separately from the Posterior).
#


# 其目的是从一个 ARK 文件中读取数据，并生成一个包含键值对 (key, cntime) 的生成器
def read_cntime_ark(file_or_fd):
    """ generator(key,vec<tuple<float,float>>) = read_cntime_ark(file_or_fd)
   Returns generator of (key,cntime) tuples, read from ark file.
   file_or_fd : file, gzipped file, pipe or opened file descriptor.

   Iterate the ark:
   for key,time in kaldi_io.read_cntime_ark(file):
     ...

   Read ark to a 'dictionary':
   d = { key:time for key,time in kaldi_io.read_post_ark(file) }
  """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            cntime = read_cntime(fd)
            yield key, cntime
            key = read_key(fd)
    finally:
        if fd is not file_or_fd: fd.close()


# 从一个文件或文件描述符中读取 Kaldi 的“混淆网络时间信息”（Confusion Network time info），这种信息是以二进制格式存储的。
def read_cntime(file_or_fd):
    """ [cntime] = read_cntime(file_or_fd)
   Reads single kaldi 'Confusion Network time info', in binary format:
   C++ type: vector<tuple<float,float> >.
   (begin/end times of bins at the confusion network).

   Binary layout is '<num-bins> <beg1> <end1> <beg2> <end2> ...'

   file_or_fd : file, gzipped file, pipe or opened file descriptor.

   Returns vector of tuples.
  """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2).decode()
    assert (binary == '\0B')
    # assuming it's binary

    assert (fd.read(1).decode() == '\4')
    # int-size
    vec_size = np.frombuffer(fd.read(4), dtype='int32',
                             count=1)[0]  # number of frames (or bins)

    data = np.frombuffer(fd.read(vec_size * 10),
                         dtype=[('size_beg', 'int8'), ('t_beg', 'float32'),
                                ('size_end', 'int8'), ('t_end', 'float32')],
                         count=vec_size)
    assert (data[0]['size_beg'] == 4)
    assert (data[0]['size_end'] == 4)
    ans = data[['t_beg',
                't_end']].tolist()  # Return vector of tuples (t_beg,t_end),

    if fd is not file_or_fd: fd.close()
    return ans


#################################################
# Segments related,
#


# Segments as 'Bool vectors' can be handy,
# - for 'superposing' the segmentations,
# - for frame-selection in Speaker-ID experiments,
# 从 Kaldi 的 segments 文件中读取数据，并将其转换为布尔向量。这个布尔向量表示每个帧是否属于某个语音段。
def read_segments_as_bool_vec(segments_file):
    """ [ bool_vec ] = read_segments_as_bool_vec(segments_file)
   using kaldi 'segments' file for 1 wav, format : '<utt> <rec> <t-beg> <t-end>'
   - t-beg, t-end is in seconds,
   - assumed 100 frames/second,
  """
    segs = np.loadtxt(segments_file, dtype='object,object,f,f', ndmin=1)
    # Sanity checks,
    assert (len(segs) > 0)  # empty segmentation is an error,
    assert (len(np.unique([rec[1] for rec in segs])) == 1
            )  # segments with only 1 wav-file,
    # Convert time to frame-indexes,
    start = np.rint([100 * rec[2] for rec in segs]).astype(int)
    end = np.rint([100 * rec[3] for rec in segs]).astype(int)
    # Taken from 'read_lab_to_bool_vec', htk.py,
    frms = np.repeat(
        np.r_[np.tile([False, True], len(end)), False],
        np.r_[np.c_[start - np.r_[0, end[:-1]], end - start].flat, 0])
    assert np.sum(end - start) == np.sum(frms)
    return frms

# 总结：kaldi文件的读取写入