from s_xbyak import *

SIMD_BYTE = 64

# expand args
# Unroll(2, op, [xm0, xm1], [xm2, xm3], xm4)
# -> op(xm0, xm2, xm4)
#    op(xm1, xm3, xm4)
def Unroll(n, op, *args, addrOffset=None):
  xs = list(args)
  for i in range(n):
    ys = []
    for e in xs:
      if isinstance(e, list):
        ys.append(e[i])
      elif isinstance(e, Address):
        if addrOffset == None:
          if e.broadcast:
            addrOffset = 0
          else:
            addrOffset = SIMD_BYTE
        ys.append(e + addrOffset*i)
      else:
        ys.append(e)
    op(*ys)

def genUnrollFunc(n):
  """
    return a function takes op and outputs a function that takes *args and outputs n unrolled op
  """
  def fn(op):
    def gn(*args, addrOffset=None):
      Unroll(n, op, *args, addrOffset=addrOffset)
    return gn
  return fn

def zipOr(v, k):
  """
    return [v[i]|v[i]]
  """
  r = []
  for i in range(len(v)):
    r.append(v[i]|k[i])
  return r

def setInt(r, v):
  mov(eax, v)
  vpbroadcastd(r, eax)

def setFloat(r, v):
  setInt(r, float2uint(v))

def cvtRegs(v, cstr):
  return list(map(lambda x:cstr(x.idx), v))

def split_n(v, n, cstr=Zmm):
  a = cvtRegs(v[:n], cstr)
  b = cvtRegs(v[n:], cstr)
  return (a, b)

def gen_fft():
  with FuncProc('fft_avx512'):
    ZN = 24 # temporary regs
    Z_CN = 4 # const regs
    with StackFrame(2, 10, vNum=ZN+Z_CN, vType=T_ZMM) as sf:
      tables = sf.p[0]
      c = sf.p[1]
      (a0, a1, a2, a3, a4, cosTbl, block, n_div_4, pim, hnn) = sf.t

      movq(a0, ptr(tables+0))

      movq(n_div_4, a0)
      shr(n_div_4, 2)
      lea(pim, ptr(c+n_div_4*8))

      (pmmp, ppmm, pmpm, perm) = sf.v[ZN:]
      vmovapd(pmmp, ptr(rip+'pmmp'))
      vmovapd(ppmm, ptr(rip+'ppmm'))
      vmovapd(pmpm, ptr(rip+'pmpm'))
      vmovapd(perm, ptr(rip+'perm'))

      def f1(n):
        un = genUnrollFunc(n)
        r = sf.v
        (v0, r) = split_n(r, n)
        (v1, r) = split_n(r, n)
        (v2, r) = split_n(r, n)
        (v3, r) = split_n(r, n)
        un(vmovapd)(v0, ptr(a1))
        un(vmovapd)(v1, ptr(a2))
        un(vshufpd)(v2, v0, v0, 0)
        un(vshufpd)(v0, v0, v0, 255)
        un(vshufpd)(v3, v1, v1, 0)
        un(vshufpd)(v1, v1, v1, 255)
        un(vfmadd231pd)(v2, pmpm, v0)
        un(vfmadd231pd)(v3, pmpm, v1)
        un(vmovapd)(ptr(a1), v2)
        un(vmovapd)(ptr(a2), v3)

      movq(a1, c)
      movq(a2, pim)
      mov(rax, n_div_4)
      shr(eax, 3)

      n=2
      align(32)
      size2L = Label()
      L(size2L)
      f1(n)

      add(a1, n*64)
      add(a2, n*64)
      sub(eax, n)
      jnz(size2L)

      def f2(n):
        un = genUnrollFunc(n)
        r = sf.v
        (v0, r) = split_n(r, n)
        (v1, r) = split_n(r, n)
        (v2, r) = split_n(r, n)
        (v3, r) = split_n(r, n)
        un(vmovapd)(v0, ptr(a1))
        un(vmovapd)(v1, ptr(a2))
        un(vmovapd)(v2, perm)
        un(vpermi2pd)(v2, v0, v1)
        un(vmovapd)(v3, perm)
        un(vpermi2pd)(v3, v1, v0)
        un(vpermpd)(v0, v0, 68)
        un(vpermpd)(v1, v1, 68)
        un(vfmadd231pd)(v0, ppmm, v2)
        un(vfmadd231pd)(v1, pmmp, v3)
        un(vmovapd)(ptr(a1), v0)
        un(vmovapd)(ptr(a2), v1)

      mov(a1, c)
      mov(a2, pim)
      mov(rax, n_div_4)
      shr(eax, 3) # n/32

      n=2
      align(32)
      size4L = Label()
      L(size4L)
      f2(n)

      add(a1, n*64)
      add(a2, n*64)
      sub(eax, n)
      jnz(size4L)

      def f3(n, cstr=Zmm):
        """
        cstr=Zmm or Ymm
        """
        simdByte = 64 if cstr == Zmm else 32
        un = genUnrollFunc(n)
        r = sf.v
        (v0, r) = split_n(r, n, cstr)
        (v1, r) = split_n(r, n, cstr)
        (v2, r) = split_n(r, n, cstr)
        (v3, r) = split_n(r, n, cstr)
        (v4, r) = split_n(r, n, cstr)
        (v5, r) = split_n(r, n, cstr)
        un(vmovapd)(v0, ptr(a3))
        un(vmovapd)(v1, ptr(a4))
        un(vmovapd)(v2, ptr(cosTbl), addrOffset=simdByte*2)
        un(vmovapd)(v3, ptr(cosTbl+simdByte), addrOffset=simdByte*2)
        un(vmulpd)(v4, v2, v0)
        un(vfnmadd231pd)(v4, v3, v1)
        un(vmulpd)(v5, v3, v0)
        un(vfmadd231pd)(v5, v2, v1)
        un(vmovapd)(v2, ptr(a1))
        un(vmovapd)(v3, ptr(a2))
        un(vsubpd)(v0, v2, v4)
        un(vmovapd)(ptr(a3), v0)
        un(vsubpd)(v1, v3, v5)
        un(vmovapd)(ptr(a4), v1)
        un(vaddpd)(v2, v2, v4)
        un(vmovapd)(ptr(a1), v2)
        un(vaddpd)(v3, v3, v5)
        un(vmovapd)(ptr(a2), v3)

      mov(a0, ptr(tables+8))

      # hnn=4 fixed
      hnn4=4
      xor_(block, block)
      mov(rax, n_div_4)
      shr(eax, 3) # rax/=hnn4*2

      align(32)
      block4L = Label()
      L(block4L)
      lea(a1, ptr(c+block*8))
      lea(a2, ptr(pim+block*8))
      lea(a3, ptr(a1+hnn4*8))
      lea(a4, ptr(a2+hnn4*8))

      mov(cosTbl, a0)

      f3(1, Ymm)

      add(block, hnn4*2)
      dec(eax)
      jnz(block4L)

      mov(hnn, 8)
      lea(a0, ptr(a0+hnn*8))
      cmp(hnn, n_div_4)

      halfL = Label()
      jb(halfL)

      mov(a0, ptr(tables+8))
      mov(hnn, 8)

      L(halfL)
      xor_(block, block)

      blockL = Label()
      align(32)
      L(blockL)
      lea(a1, ptr(c+block*8))
      lea(a2, ptr(pim+block*8))
      lea(a3, ptr(a1+hnn*8))
      lea(a4, ptr(a2+hnn*8))
      mov(cosTbl, a0)
      mov(rax, hnn)
      shr(eax, 3)

      hnn_exitL = Label()
      mul4L = Label()
      test(eax, 3)
      jz(mul4L)

      # eax<=3
      align(32)
      mod3L = Label()
      L(mod3L)
      n=1
      f3(n, Zmm)

      add(a1, n*64)
      add(a2, n*64)
      add(a3, n*64)
      add(a4, n*64)
      add(cosTbl, n*128)
      sub(eax, n)
      and_(eax, 3)
      jnz(mod3L)

      test(eax, eax)
      jz(hnn_exitL)

      # hnn_even
      n=4
      align(32)
      L(mul4L)
      f3(n, Zmm)

      add(a1, n*64)
      add(a2, n*64)
      add(a3, n*64)
      add(a4, n*64)
      add(cosTbl, n*128)
      sub(eax, n)
      jnz(mul4L)

      L(hnn_exitL)

      lea(block, ptr(block+hnn*2))
      cmp(block, n_div_4)
      jnz(blockL)

      shl(hnn, 1)
      lea(a0, ptr(a0+hnn*8))
      cmp(hnn, n_div_4)
      jnz(halfL)

      def f4(n):
        un = genUnrollFunc(n)
        r = sf.v
        (v0, r) = split_n(r, n)
        (v1, r) = split_n(r, n)
        (v2, r) = split_n(r, n)
        (v3, r) = split_n(r, n)
        (w0, r) = split_n(r, n)
        (w1, r) = split_n(r, n)
        """
        (v0)=(v2 -v3)(v0)
        (v1) (v3  v2)(v1)
        """
        un(vmovapd)(v0, ptr(a1))
        un(vmovapd)(v1, ptr(a2))
        un(vmovapd)(v2, ptr(a0), addrOffset=SIMD_BYTE*2)
        un(vmovapd)(v3, ptr(a0+SIMD_BYTE), addrOffset=SIMD_BYTE*2)

        un(vmulpd)(w0, v1, v2)
        un(vmulpd)(w1, v1, v3)
        un(vfmadd132pd)(v3, w0, v0)
        un(vfmsub132pd)(v2, w1, v0)
        un(vmovapd)(ptr(a1), v2)
        un(vmovapd)(ptr(a2), v3)

      mov(a1, c)
      mov(a2, pim)
      mov(rax, n_div_4)
      shr(eax, 3) # n/32

      # assume(n >= 128)
      n=2
      finalL = Label()
      L(finalL)
      f4(n)

      add(a1, n*64)
      add(a2, n*64)
      add(a0, n*128)
      sub(eax, n)
      jnz(finalL)


def main():
  parser = getDefaultParser()
  param = parser.parse_args()

  init(param)
  segment('data')
  align(64)
  tbl = [('pmmp', (+1, -1, -1, +1, +1, -1, -1, +1)),
         ('ppmm', (+1, +1, -1, -1, +1, +1, -1, -1)),
         ('pmpm', (+1, -1, +1, -1, +1, -1, +1, -1))]
  for (name, v) in tbl:
    makeLabel(name)
    for e in v:
      dq_(double2uint(e))
  makeLabel('perm')
  for x in [2, 8+3, 2, 8+3, 6, 8+7, 6, 8+7]:
    dq_(x)

  segment('text')
  gen_fft()

  term()

if __name__ == '__main__':
  main()
