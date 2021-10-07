;; NOTE: Assertions have been generated by update_lit_checks.py --all-items and should not be edited.
;; NOTE: This test was ported using port_test.py and could be cleaned up.

;; RUN: wasm-opt %s --optimize-instructions --nominal --roundtrip --all-features -S -o - | filecheck %s
;; roundtrip to see the effects on heap types in the binary format, specifically
;; regarding nominal heap types

(module
 ;; Create multiple different signature types, all identical structurally but
 ;; distinct nominally. The three tables will use different ones, and the
 ;; emitted call_indirects should use the corresponding ones.

 ;; CHECK:      (type $v1 (func))
 (type $v1 (func))

 ;; CHECK:      (type $v2 (func))
 (type $v2 (func))

 ;; CHECK:      (type $v3 (func))
 (type $v3 (func))

 ;; CHECK:      (type $none_=>_none (func))

 ;; CHECK:      (type $i32_=>_none (func (param i32)))

 ;; CHECK:      (table $table-1 10 (ref null $v1))
 (table $table-1 10 (ref null $v1))

 ;; CHECK:      (table $table-2 10 (ref null $v2))
 (table $table-2 10 (ref null $v2))

 ;; CHECK:      (table $table-3 10 (ref null $v3))
 (table $table-3 10 (ref null $v3))

 ;; CHECK:      (elem $elem-1 (table $table-1) (i32.const 0) (ref null $v1) (ref.func $helper-1))
 (elem $elem-1 (table $table-1) (i32.const 0) (ref null $v1)
  (ref.func $helper-1))

 ;; CHECK:      (elem $elem-2 (table $table-2) (i32.const 0) (ref null $v2) (ref.func $helper-2))
 (elem $elem-2 (table $table-2) (i32.const 0) (ref null $v2)
  (ref.func $helper-2))

 ;; CHECK:      (elem $elem-3 (table $table-3) (i32.const 0) (ref null $v3) (ref.func $helper-3))
 (elem $elem-3 (table $table-3) (i32.const 0) (ref null $v3)
  (ref.func $helper-3))

 ;; CHECK:      (func $helper-1
 ;; CHECK-NEXT:  (nop)
 ;; CHECK-NEXT: )
 (func $helper-1 (type $v1))
 ;; CHECK:      (func $helper-2
 ;; CHECK-NEXT:  (nop)
 ;; CHECK-NEXT: )
 (func $helper-2 (type $v2))
 ;; CHECK:      (func $helper-3
 ;; CHECK-NEXT:  (nop)
 ;; CHECK-NEXT: )
 (func $helper-3 (type $v3))

 ;; CHECK:      (func $call-table-get (param $x i32)
 ;; CHECK-NEXT:  (call_indirect $table-1 (type $v1)
 ;; CHECK-NEXT:   (local.get $x)
 ;; CHECK-NEXT:  )
 ;; CHECK-NEXT:  (call_indirect $table-2 (type $v2)
 ;; CHECK-NEXT:   (local.get $x)
 ;; CHECK-NEXT:  )
 ;; CHECK-NEXT:  (call_indirect $table-3 (type $v3)
 ;; CHECK-NEXT:   (local.get $x)
 ;; CHECK-NEXT:  )
 ;; CHECK-NEXT: )
 (func $call-table-get (param $x i32)
  ;; The heap type of the call_indirects that we emit here should be the
  ;; identical one as on the table that they correspond to.
  (call_ref
   (table.get $table-1
    (local.get $x)
   )
  )
  (call_ref
   (table.get $table-2
    (local.get $x)
   )
  )
  (call_ref
   (table.get $table-3
    (local.get $x)
   )
  )
 )
)
