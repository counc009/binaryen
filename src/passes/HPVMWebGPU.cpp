//
// Perform translation from WASM to WebGPU
//

#include "ir/module-utils.h"
#include "ir/utils.h"
#include "pass.h"
#include "wasm.h"
#include "wasm-builder.h"

#include <map>
#include <stack>
#include <string>

namespace wasm {

static bool findKernel(Module* module, std::string kernelName, Name& funcName) {
  for (auto& exp : module->exports) {
    Name name = exp->name;
    Name value = exp->value;
    if (std::string(name.c_str()) == kernelName) {
      funcName = value;
      return true;
    }
  }
  return false;
}

static void tokenize(std::string const& str, const char delim,
                     std::vector<std::string>& out) {
  size_t start, end = 0;
  while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
    end = str.find(delim, start);
    out.push_back(str.substr(start, end-start));
  }
}

struct PassedType {
  std::string type;
  bool idx, dim, write;

  PassedType(std::string ty, bool i, bool d, bool w)
    : type(ty), idx(i), dim(d), write(w) {}
};

static bool extractTypes(std::string const& str, std::vector<PassedType>& out) {
  std::vector<std::string> tokens;
  tokenize(str, ',', tokens);
  for (const std::string& token : tokens) {
    if (token[1] != ':') {
      std::cerr << "Argument type not specified in form <access>:<type>\n";
      return false;
    }
    out.push_back({token.substr(2), token[0] == 'i', token[0] == 'd',
                   token[0] == 'w'});
  }
  return true;
}

enum InferType : int32_t {
  i32, u32, i64, u64, f32, f64,
  i32_ptr, u32_ptr, i64_ptr, u64_ptr, f32_ptr, f64_ptr,
  x_idx, y_idx, z_idx, x_dim, y_dim, z_dim,
  n32, n64, /* types for integers that we know the size of, but not sign */
  noType /* for blocks, loops, etc. */
};

static bool isArray(InferType ty) {
  switch (ty) {
    case InferType::i32_ptr: case InferType::u32_ptr: case InferType::i64_ptr:
    case InferType::u64_ptr: case InferType::f32_ptr: case InferType::f64_ptr:
      return true;
    default: return false;
  }
}

static InferType getElemType(InferType ty) {
  switch (ty) {
    case InferType::i32_ptr: return InferType::i32;
    case InferType::u32_ptr: return InferType::u32;
    case InferType::i64_ptr: return InferType::i64;
    case InferType::u64_ptr: return InferType::u64;
    case InferType::f32_ptr: return InferType::f32;
    case InferType::f64_ptr: return InferType::f64;
    default: return InferType::noType;
  }
}

static int getBytes(InferType ty) {
  switch (ty) {
    case InferType::i32: case InferType::u32: case InferType::f32: return 4;
    case InferType::i64: case InferType::u64: case InferType::f64: return 8;
    // Only need this to work for possible array element types
    default: return 0;
  }
}

static InferType typeForEquality(InferType ty) {
  switch (ty) {
    case InferType::x_idx: case InferType::y_idx: case InferType::z_idx:
    case InferType::x_dim: case InferType::y_dim: case InferType::z_dim:
      return InferType::u32;
    default:
      return ty;
  }
}

static std::string typeToWebGPU(InferType ty) {
  switch (ty) {
    case InferType::i32: return "i32";
    case InferType::u32: return "u32";
    case InferType::i64: return "i64";
    case InferType::u64: return "u64";
    case InferType::f32: return "f32";
    case InferType::f64: return "f64";
    case InferType::i32_ptr: return "array<i32>";
    case InferType::u32_ptr: return "array<u32>";
    case InferType::i64_ptr: return "array<i64>";
    case InferType::u64_ptr: return "array<u64>";
    case InferType::f32_ptr: return "array<f32>";
    case InferType::f64_ptr: return "array<f64>";
    case InferType::x_idx: return "u32";
    case InferType::y_idx: return "u32";
    case InferType::z_idx: return "u32";
    case InferType::x_dim: return "u32";
    case InferType::y_dim: return "u32";
    case InferType::z_dim: return "u32";

    case InferType::n32:
      std::cerr << "Converting an unknown signed n32 to WebGPU type\n";
      return "i32";
    case InferType::n64:
      std::cerr << "Converting an unknown signed n64 to WebGPU type\n";
      return "i64";
    case InferType::noType:
      std::cerr << "Converting a none type to WebGPU type\n";
      return "none";
  }
}

#define CASE_TYPE(wasmType, cases)                                             \
    case Type::BasicType::wasmType:                                            \
      if (0) ;                                                                 \
      cases                                                                    \
      else {                                                                   \
        std::cerr << "Function has argument type " #wasmType                   \
                  << ", but provided type of " << types[i].type                \
                  << " does not match, not supported (" << __FILE__            \
                  << " : " << __LINE__ << ")\n";                               \
        return false;                                                          \
      }                                                                        \
      break;
#define PROVIDED(ty)                                                           \
  else if (types[i].type == #ty) variableTypes.push_back(InferType::ty);
#define PROVIDED_PTR(ety)                                                      \
  else if (types[i].type == #ety "*") variableTypes.push_back(InferType::ety##_ptr);
#define PROVIDED_IDX(which)                                                    \
  else if (types[i].type == #which && types[i].idx)                            \
    variableTypes.push_back(InferType::which##_idx);
#define PROVIDED_DIM(which)                                                    \
  else if (types[i].type == #which && types[i].dim)                            \
    variableTypes.push_back(InferType::which##_dim);
#define ASSIGN(ty)                                                             \
  else if (1) variableTypes.push_back(InferType::ty);

#define LOCATE_IDXDIM(which, pos, idx)                                         \
  case InferType::pos##_##which:                                               \
    if (which[idx] != -1) {                                                    \
      std::cerr << "Multiple arguments for index " #pos "\n";                  \
      return false;                                                            \
    }                                                                          \
    which[idx] = i;                                                            \
    break;
#define VERIFY_IDXDIM(which, pos, idx)                                         \
  if (which[idx] == -1) {                                                      \
    std::cerr << "Missing argument for index " #pos "\n";                      \
    return false;                                                              \
  }

static bool locateIdxDim(std::vector<InferType> const& types,
                         int* idx, int* dim) {
  for (unsigned int i = 0; i < types.size(); i++) {
    switch (types[i]) {
      LOCATE_IDXDIM(idx, x, 0)
      LOCATE_IDXDIM(idx, y, 1)
      LOCATE_IDXDIM(idx, z, 2)
      LOCATE_IDXDIM(dim, x, 0)
      LOCATE_IDXDIM(dim, y, 1)
      LOCATE_IDXDIM(dim, z, 2)
      default: break;
    }
  }

  VERIFY_IDXDIM(idx, x, 0)
  VERIFY_IDXDIM(idx, y, 1)
  VERIFY_IDXDIM(idx, z, 2)
  VERIFY_IDXDIM(dim, x, 0)
  VERIFY_IDXDIM(dim, y, 1)
  VERIFY_IDXDIM(dim, z, 2)

  return true;
}

#define EXPR_CASE(tag) case Expression::Id::tag:

static unsigned int uniq() { static unsigned int n = 0; return ++n; }

struct CFGNode {
  unsigned int index;
  std::vector<Expression*> body;
  bool conditional;
  Expression* cond;
  std::vector<CFGNode*> succs; // First will represent if condition is true
  std::vector<CFGNode*> preds;
};

CFGNode* createNode(std::vector<CFGNode*>& nodes) {
  CFGNode* result = new CFGNode;
  nodes.push_back(result);
  result->index = nodes.size() - 1;
  return result;
}

CFGNode* splitNode(CFGNode* node, std::vector<CFGNode*>& nodes) {
  CFGNode* newNode = createNode(nodes);
  newNode->conditional = node->conditional;
  newNode->cond = node->cond;
  newNode->succs = node->succs;
  newNode->preds = {node};

  node->conditional = false;
  node->cond = nullptr;
  node->succs = {newNode};

  return newNode;
}

CFGNode deadBranch;

static bool onlyIfCF(Expression* expr) {
  switch (expr->_id) {
    EXPR_CASE(BlockId) EXPR_CASE(LoopId) EXPR_CASE(BreakId) return false;
    EXPR_CASE(LocalGetId) EXPR_CASE(LocalSetId) EXPR_CASE(ConstId)
      EXPR_CASE(NopId) return true;
    EXPR_CASE(IfId) {
      If* ifExp = static_cast<If*>(expr);
      return onlyIfCF(ifExp->condition) && onlyIfCF(ifExp->ifTrue)
          && onlyIfCF(ifExp->ifFalse);
    }
    EXPR_CASE(LoadId) {
      Load* loadExp = static_cast<Load*>(expr);
      return onlyIfCF(loadExp->ptr);
    }
    EXPR_CASE(StoreId) {
      Store* storeExp = static_cast<Store*>(expr);
      return onlyIfCF(storeExp->ptr) && onlyIfCF(storeExp->value);
    }
    EXPR_CASE(UnaryId) {
      Unary* unaryExp = static_cast<Unary*>(expr);
      return onlyIfCF(unaryExp->value);
    }
    EXPR_CASE(BinaryId) {
      Binary* binaryExp = static_cast<Binary*>(expr);
      return onlyIfCF(binaryExp->left) && onlyIfCF(binaryExp->right);
    }
    default: {
      std::cerr << "Encountered unhandled expression (" __FILE__ " : "
                << __LINE__ << "):\n"; expr->dump();
      return false;
    }
  }
}

// Returns pointer to the node for code that goes after the processed
// expression. Returns NULL if there's an error.
static CFGNode* constructCFG(Expression* expr, CFGNode* curNode,
                             std::map<Name, CFGNode*>& branches,
                             std::vector<CFGNode*>& nodes) {
  if (curNode == &deadBranch) {
    std::cerr << "Cannot construct CFG to the dead node. Inst:\n";
    expr->dump();
    return nullptr;
  }
  switch (expr->_id) {
    EXPR_CASE(BlockId) {
      Block* block = static_cast<Block*>(expr);
      Name blockName = block->name;

      if (blockName.str) {
        if (branches.find(blockName) != branches.end()) {
          std::cerr << "Block label already exists: " << blockName << "\n";
          return nullptr;
        }
        CFGNode* blockEnd = splitNode(curNode, nodes);
        branches[blockName] = blockEnd;

        for (Expression* ex : block->list) {
          if (!curNode) break;
          curNode = constructCFG(ex, curNode, branches, nodes);
        }

        branches.erase(branches.find(blockName));
        return blockEnd;
      } else {
        for (Expression* ex : block->list) {
          if (!curNode) break;
          curNode = constructCFG(ex, curNode, branches, nodes);
        }

        return curNode;
      }
    }
    EXPR_CASE(IfId) {
      If* ifExp = static_cast<If*>(expr);
      Expression* cond = ifExp->condition;
      Expression* ifTrue = ifExp->ifTrue;
      Expression* ifElse = ifExp->ifFalse;

      CFGNode* nodeJoin = splitNode(curNode, nodes);
      CFGNode* nodeThen = createNode(nodes);
      CFGNode* nodeElse = createNode(nodes);

      curNode->conditional = true;
      curNode->cond = cond;
      curNode->succs = {nodeThen, nodeElse};

      nodeThen->conditional = false;
      nodeThen->succs = {nodeJoin};
      nodeThen->preds = {curNode};

      nodeElse->conditional = false;
      nodeElse->succs = {nodeJoin};
      nodeThen->preds = {curNode};

      nodeJoin->preds = {nodeThen, nodeElse};

      if (!constructCFG(ifTrue, nodeThen, branches, nodes)) return nullptr;
      if (!constructCFG(ifElse, nodeElse, branches, nodes)) return nullptr;

      return nodeJoin;
    }
    EXPR_CASE(LoopId) {
      Loop* loopExp = static_cast<Loop*>(expr);
      Name loopName = loopExp->name;
      Expression* body = loopExp->body;

      if (branches.find(loopName) != branches.end()) {
        std::cerr << "Loop label already exists: " << loopName << "\n";
        return nullptr;
      }
      CFGNode* loopNode = splitNode(curNode, nodes);
      branches[loopName] = loopNode;

      CFGNode* result = constructCFG(body, loopNode, branches, nodes);

      branches.erase(branches.find(loopName));
      return result;
    }
    EXPR_CASE(BreakId) {
      Break* breakExp = static_cast<Break*>(expr);
      Name toName = breakExp->name;
      Expression* val = breakExp->value;
      Expression* cond = breakExp->condition;

      if (branches.find(toName) == branches.end()) {
        std::cerr << "Branch to undeclared branch\n";
        return nullptr;
      }
      CFGNode* branchTo = branches[toName];

      if (val) {
        std::cerr << "Break with value is not currently supported (" __FILE__
                     " : " << __LINE__ << ")\n";
        breakExp->dump();
        return nullptr;
      } else if (!cond) {
        // Unconditional Branch
        curNode->conditional = false;
        curNode->cond = nullptr;
        curNode->succs = {branchTo};
        branchTo->preds.push_back(curNode);
        return &deadBranch;
      } else {
        // Conditional Branch
        CFGNode* followBlock = splitNode(curNode, nodes);
        curNode->conditional = true;
        curNode->cond = cond;
        curNode->succs = {branchTo, followBlock};
        branchTo->preds.push_back(curNode);
        return followBlock;
      }
    }
    EXPR_CASE(LocalGetId)
    EXPR_CASE(LocalSetId)
    EXPR_CASE(ConstId)
    EXPR_CASE(NopId) {
      curNode->body.push_back(expr);
      return curNode;
    }
    EXPR_CASE(LoadId) {
      Load* loadExp = static_cast<Load*>(expr);
      if (onlyIfCF(loadExp->ptr)) {
        curNode->body.push_back(expr);
        return curNode;
      } else {
        std::cerr << "Control-flow beyond if-then-else is not supported in an "
                     "expression (" __FILE__ " : " << __LINE__ << ")\n";
        return nullptr;
      }
    }
    EXPR_CASE(StoreId) {
      Store* storeExp = static_cast<Store*>(expr);
      if (onlyIfCF(storeExp->ptr) && onlyIfCF(storeExp->value)) {
        curNode->body.push_back(expr);
        return curNode;
      } else {
        std::cerr << "Control-flow beyond if-then-else is not supported in an "
                     "expression (" __FILE__ " : " << __LINE__ << ")\n";
        return nullptr;
      }
    }
    EXPR_CASE(UnaryId) {
      Unary* unaryExp = static_cast<Unary*>(expr);
      if (onlyIfCF(unaryExp->value)) {
        curNode->body.push_back(expr);
        return curNode;
      } else {
        std::cerr << "Control-flow beyond if-then-else is not supported in an "
                     "expression (" __FILE__ " : " << __LINE__ << ")\n";
        return nullptr;
      }
    }
    EXPR_CASE(BinaryId) {
      Binary* binaryExp = static_cast<Binary*>(expr);
      if (onlyIfCF(binaryExp->left) && onlyIfCF(binaryExp->right)) {
        curNode->body.push_back(expr);
        return curNode;
      } else {
        std::cerr << "Control-flow beyond if-then-else is not supported in an "
                     "expression (" __FILE__ " : " << __LINE__ << ")\n";
        return nullptr;
      }
    }
    default: {
      std::cerr << "Encountered unhandled expression (" __FILE__ " : "
                << __LINE__ << "):\n"; expr->dump();
      return nullptr;
    }
  }
}

static bool constructCFG(Expression* expr, std::vector<CFGNode*>& nodes) {
  std::map<Name, CFGNode*> branches;
  CFGNode* start = createNode(nodes);
  start->conditional = false;
  start->succs = {nullptr};
  start->preds = {};

  return constructCFG(expr, start, branches, nodes) != nullptr;
}

static bool emitCode(Expression* expr, std::string& output,
                     std::vector<InferType>& types,
                     std::string* valName=nullptr,
                     InferType* resType=nullptr);

static bool emitArray(Expression* expr, std::string& output,
                      std::vector<InferType>& types,
                      std::string& idxName, std::string& arrayName,
                      InferType& elemType) {
  if (expr->_id != Expression::Id::BinaryId) {
    std::cerr << "Failed to analyze array access, top level is not a binary "
                  "operation:\n";
    expr->dump();
    return false;
  }

  Binary* binaryExp = static_cast<Binary*>(expr);
  if (binaryExp->op != BinaryOp::AddInt32) {
    std::cerr << "Failed to analyze array access, top level is not an add:\n";
    expr->dump();
    return false;
  }

  Expression* left = binaryExp->left;
  Expression* right = binaryExp->right;

  bool leftArray = false, rightArray = false; Index arrayIndex;
  InferType arrayElemType;
  if (left->_id == Expression::Id::LocalGetId) {
    Index access = static_cast<LocalGet*>(left)->index;
    if (isArray(types[access])) {
      leftArray = true;
      arrayIndex = access;
      arrayElemType = getElemType(types[access]);
    }
  }
  if (right->_id == Expression::Id::LocalGetId) {
    Index access = static_cast<LocalGet*>(right)->index;
    if (isArray(types[access])) {
      rightArray = true;
      arrayIndex = access;
      arrayElemType = getElemType(types[access]);
    }
  }
  if (!leftArray && !rightArray) {
    std::cerr << "Failed to analyze array access, neither operands of top "
                 "level add is an array:\n";
    expr->dump();
    return false;
  } else if (leftArray && rightArray) {
    std::cerr << "Failed to analyze array access, both operands of top level "
                 "add are arrays:\n";
    expr->dump();
    return false;
  }

  std::string indexName; InferType indexType;
  if (!emitCode(leftArray ? right : left, output, types, &indexName, &indexType))
    return false;
  if (indexType == InferType::noType || indexName == "") {
    std::cerr << "Index expression did not produce a value:\n";
    (leftArray ? right : left)->dump();
    return false;
  }

  std::string idx = "idx" + std::to_string(uniq());
  output += "let " + idx + " = " + indexName + " / "
          + typeToWebGPU(indexType) + "("
          + std::to_string(getBytes(arrayElemType)) + ");\n";

  arrayName = "arg" + std::to_string(arrayIndex);
  idxName = idx;
  elemType = arrayElemType;
  return true;
}

#define GEN_BINARY_OP(OpName, Operator)                                        \
  case BinaryOp::OpName: {                                                     \
    output += "let " + tmp + " = " + typeToWebGPU(opType)                      \
            + "(" + leftName + ") " Operator " " + typeToWebGPU(opType)        \
            + "(" + rightName + ");\n";                                        \
    if (valName) *valName = tmp;                                               \
    if (resType) *resType = opType;                                            \
    break;                                                                     \
  }
#define GEN_BINARY_OP_ALL_TYPES(OpName, Operator)                              \
  GEN_BINARY_OP(OpName##Int32, Operator)                                       \
  GEN_BINARY_OP(OpName##Int64, Operator)                                       \
  GEN_BINARY_OP(OpName##Float32, Operator)                                     \
  GEN_BINARY_OP(OpName##Float64, Operator)

static bool emitCode(Expression* expr, std::string& output,
                     std::vector<InferType>& types,
                     std::string* valName, InferType* resType) {
  if (valName) *valName = "";
  if (resType) *resType = InferType::noType;

  switch (expr->_id) {
    EXPR_CASE(IfId) { // Must be if-expressions
      std::cerr << "If-expressions currently not supported (" __FILE__ " : "
                << __LINE__ << ")\n";
      return false;
    }
    EXPR_CASE(LocalGetId) {
      LocalGet* getExp = static_cast<LocalGet*>(expr);
      Index local = getExp->index;
      if (valName) *valName = "local" + std::to_string(local);
      if (resType) *resType = types[local];
      break;
    }
    EXPR_CASE(LocalSetId) {
      LocalSet* setExp = static_cast<LocalSet*>(expr);
      Index local = setExp->index;
      Expression* value = setExp->value;
      std::string valueName; InferType valueType;
      if (!emitCode(value, output, types, &valueName, &valueType))
        return false;
      if (valueType == InferType::noType || valueName == "") {
        std::cerr << "Assignment's value did not produce a value ("
                     __FILE__ " : " << __LINE__ << ")\n";
        return false;
      }
      if (types[local] == InferType::n32 || types[local] == InferType::n64)
        types[local] = valueType;

      if (isArray(valueType)) {
        std::cerr << "Setting a local to be an array-type is not supported ("
                     __FILE__ " : " << __LINE__ << ")\n";
        return false;
      }
      if (typeForEquality(types[local]) == typeForEquality(valueType)) {
        output += "local" + std::to_string(local) + " = " + valueName + ";\n";
      } else {
        output += "local" + std::to_string(local) + " = "
                + typeToWebGPU(types[local]) + "(" + valueName + ");\n";
      }
      if (setExp->isTee()) {
        if (valName) *valName = "local" + std::to_string(local);
        if (resType) *resType = types[local];
      }
      break;
    }
    EXPR_CASE(NopId) {
      break; // Don't need to do anything on a NOP
    }
    EXPR_CASE(LoadId) {
      Load* loadExp = static_cast<Load*>(expr);

      std::string idxName, arrayName; InferType elemType;
      if (!emitArray(loadExp->ptr, output, types, idxName, arrayName,
                     elemType))
        return false;

      std::string tmp = "tmp" + std::to_string(uniq());
      output += "let " + tmp + " = " + arrayName + ".value[" + idxName + "];\n";
      if (valName) *valName = tmp;
      if (resType) *resType = elemType;
      break;
    }
    EXPR_CASE(StoreId) {
      Store* storeExp = static_cast<Store*>(expr);

      std::string idxName, arrayName; InferType elemType;
      if (!emitArray(storeExp->ptr, output, types, idxName, arrayName,
                     elemType))
        return false;

      std::string valueName; InferType valueType;
      if (!emitCode(storeExp->value, output, types, &valueName, &valueType))
        return false;

      if (valueType == InferType::noType || valueName == "") {
        std::cerr << "Store's value did not produce a value (" __FILE__ " : "
                  << __LINE__ << ")\n";
        return false;
      }

      if (typeToWebGPU(valueType) != typeToWebGPU(elemType))
        output += arrayName + ".value[" + idxName + "] = "
                + typeToWebGPU(elemType) + "(" + valueName + ");\n";
      else
        output += arrayName + ".value[" + idxName + "] = " + valueName + ";\n";
      break;
    }

    EXPR_CASE(ConstId) {
      Literal value = static_cast<Const*>(expr)->value;
      if (!value.type.isBasic()) {
        std::cerr << "Constant non-basic type not supported (" __FILE__
                     " : " << __LINE__ << ")\n";
        return false;
      }
      std::string tmp = "tmp" + std::to_string(uniq());
      switch (value.type.getBasic()) {
        case Type::BasicType::i32: {
          output += "let " + tmp + " : i32 = "
                  + std::to_string(value.geti32()) + ";\n";
          if (valName) *valName = tmp;
          if (resType) *resType = InferType::n32;
          break;
        }
        case Type::BasicType::i64: {
          output += "let " + tmp + " : i64 = "
                  + std::to_string(value.geti64()) + ";\n";
          if (valName) *valName = tmp;
          if (resType) *resType = InferType::n64;
          break;
        }
        case Type::BasicType::f32: {
          output += "let " + tmp + " : f32 = "
                  + std::to_string(value.getf32()) + ";\n";
          if (valName) *valName = tmp;
          if (resType) *resType = InferType::f32;
          break;
        }
        case Type::BasicType::f64: {
          output += "let " + tmp + " : f64 = "
                  + std::to_string(value.getf64()) + ";\n";
          if (valName) *valName = tmp;
          if (resType) *resType = InferType::f64;
          break;
        }
        default:
          std::cerr << "Constant basic type not supported (" __FILE__ " : "
                    << __LINE__ << "\n";
          return false;
      }
      break;
    }

    EXPR_CASE(UnaryId) {
      Unary* unaryExp = static_cast<Unary*>(expr);

      std::string valueName; InferType valueType;
      if (!emitCode(unaryExp->value, output, types, &valueName, &valueType))
        return false;

      if (valueType == InferType::noType || valueName == "") {
        std::cerr << "UnaryExpr's value did not produce a value (" __FILE__
                     " : " << __LINE__ << ")\n";
        return false;
      }

      //std::string tmp = "tmp" + std::to_string(uniq());
      switch (unaryExp->op) {
        default:
          std::cerr << "Unsupported unary operation encountered (" __FILE__
                       " : " << __LINE__ << "):\n";
          expr->dump();
          return false;
      }
      break;
    }
    EXPR_CASE(BinaryId) {
      Binary* binaryExp = static_cast<Binary*>(expr);

      std::string leftName, rightName; InferType leftType, rightType;
      if (!emitCode(binaryExp->left, output, types, &leftName, &leftType))
        return false;
      if (!emitCode(binaryExp->right, output, types, &rightName, &rightType))
        return false;

      if (leftName == "" || leftType == InferType::noType || rightName == ""
          || rightType == InferType::noType) {
        std::cerr << "Binary expression operand did not produce a value ("
                     __FILE__ " : " << __LINE__ << ")\n";
        binaryExp->dump();
        return false;
      }

      InferType opType = InferType::noType;
      if (typeForEquality(leftType) == typeForEquality(rightType))
        opType = leftType;
      else if (leftType == InferType::n32 || leftType == InferType::n64)
        opType = rightType;
      else if (rightType == InferType::n32 || rightType == InferType::n64)
        opType = leftType;
      else {
        std::cerr << "Type mistach in binary expression (" __FILE__ " : "
                  << __LINE__ << ")\n";
        std::cerr << "Left " << typeToWebGPU(leftType) << ", Right "
                  << typeToWebGPU(rightType) << "\n";
        return false;
      }

      std::string tmp = "tmp" + std::to_string(uniq());
      switch (binaryExp->op) {
        GEN_BINARY_OP_ALL_TYPES(Add, "+")
        GEN_BINARY_OP_ALL_TYPES(Sub, "-")
        GEN_BINARY_OP_ALL_TYPES(Mul, "*")
        GEN_BINARY_OP_ALL_TYPES(Eq, "==")
        GEN_BINARY_OP_ALL_TYPES(Ne, "!=")
        case BinaryOp::ShlInt32: case BinaryOp::ShlInt64: {
          // NOTE: The WebGPU WGSL standard says shifts are performed using a
          // shiftLeft(x, y) operator, but Firefox seems to not support this
          // and instead supports x << y
          output += "let " + tmp + " = "
                  + typeToWebGPU(opType) + "(" + leftName + ") << "
                  + typeToWebGPU(opType) + "(" + rightName + ");\n";
          if (valName) *valName = tmp;
          if (resType) *resType = opType;
          break;
        }
        default:
          std::cerr << "Unsupported binary operation encountered (" __FILE__
                       " : " << __LINE__ << "):\n";
          expr->dump();
          return false;
      }
      break;
    }

    default:
      std::cerr << "Encountered unhandled expression (" __FILE__ " : "
                << __LINE__ << "): " << expr->_id << "\n"; expr->dump();
      return false;
  }
  return true;
}

static bool emitCode(CFGNode const& node, std::string& output,
                     std::vector<InferType>& types) {
  for (Expression* exp : node.body) {
    if (!emitCode(exp, output, types)) return false;
  }
  return true;
}

static void computeDominances(std::vector<CFGNode*>& cfg, CFGNode* start, CFGNode* end,
    std::vector<std::pair<std::set<CFGNode*>, std::set<CFGNode*>>>& dominances) {

  using setPair = std::pair<std::set<CFGNode*>, std::set<CFGNode*>>;
  using setFunc = std::function<std::set<CFGNode*>*(setPair&)>;
  using predFunc = std::function<std::vector<CFGNode*>*(CFGNode*)>;

  const unsigned int size = cfg.size();
  const std::set<CFGNode*> allNodes(cfg.begin(), cfg.end());

  auto computeDominance = [&](setFunc getSet, predFunc getPred,
                              CFGNode* root) {
    *getSet(dominances[root->index]) = std::set<CFGNode*>({root});
    const unsigned int rootIndex = root->index;
    
    for (unsigned int i = 0; i < size; i++)
      if (i != rootIndex)
        *getSet(dominances[i]) = allNodes;

    bool changed;
    do {
      changed = false;
      for (unsigned int i = 0; i < size; i++) {
        if (i != rootIndex) {
          CFGNode* node = cfg[i];
          std::vector<CFGNode*> const& preds = *getPred(node);
          std::set<CFGNode*> newDoms = *getSet(dominances[preds[0]->index]);
          const unsigned int predsSize = preds.size();
          for (unsigned int j = 1; j < predsSize; j++) {
            std::set<CFGNode*> tmp;
            std::set<CFGNode*> domPreds = *getSet(dominances[preds[j]->index]);
            std::set_intersection(newDoms.begin(), newDoms.end(),
                                  domPreds.begin(), domPreds.end(),
                                  std::inserter(tmp, tmp.begin()));
            newDoms = tmp;
          }
          newDoms.insert(node);
          if (newDoms != *getSet(dominances[node->index])) {
            *getSet(dominances[node->index]) = newDoms;
            changed = true;
          }
        }
      }
    } while (changed);
  };

  setFunc pairFirst  = [](setPair& s){ return &(s.first); };
  setFunc pairSecond = [](setPair& s){ return &(s.second); };
  predFunc nodePreds = [](CFGNode* n){ return &(n->preds); };
  predFunc nodeSuccs = [](CFGNode* n){ return &(n->succs); };
  computeDominance(pairFirst, nodePreds, start);
  computeDominance(pairSecond, nodeSuccs, end);
}

static bool computeBackEdges(std::vector<CFGNode*>& cfg,
    std::vector<std::pair<std::set<CFGNode*>, std::set<CFGNode*>>>& dominances,
    std::vector<std::pair<CFGNode*, CFGNode*>>& backEdges) {
  for (CFGNode* node : cfg) {
    int numBackEdges = 0;
    for (CFGNode* succ : node->succs) {
      std::set<CFGNode*>& dominators = dominances[node->index].first;
      auto f = dominators.find(succ);
      if (f != dominators.end()) {
        numBackEdges++;
        backEdges.push_back(std::make_pair(node, succ));
      }
    }
    if (numBackEdges > 1) {
      std::cerr << "Node with two back-edges out is not supported (" __FILE__
                   " : " << __LINE__ << ")\n";
      return false;
    }
  }
  return true;
}

static bool computeLoops(std::vector<std::pair<CFGNode*, CFGNode*>>& backEdges,
    // Maps header to the unique exit block (the block after the loop) and
    // the set of nodes in the natural loop
    std::map<CFGNode*, std::pair<CFGNode*, std::set<CFGNode*>>>& loops) {
  for (std::pair<CFGNode*, CFGNode*> edge : backEdges) {
    CFGNode* header = edge.second;
    CFGNode* node = edge.first;
    CFGNode* exit = !(node->conditional) ? nullptr
                  : (node->succs[0] == header ? node->succs[1]
                                              : node->succs[0]);

    auto f = loops.find(header);
    if (f == loops.end()) {
      loops[header] = std::make_pair(exit, std::set<CFGNode*>());
      f = loops.find(header);
    } else {
      CFGNode* prevExit = f->second.first;
      if (prevExit && exit && prevExit != exit) {
        std::cerr << "Loop has multiple exit blocks, not supported (" __FILE__
                     " : " << __LINE__ << ")\n";
        return false;
      } else if (!prevExit) {
        f->second.first = exit;
      }
    }

    std::set<CFGNode*>& body = f->second.second;
    body.insert(header);

    std::stack<CFGNode*> stack;
    stack.push(node);
    while (!stack.empty()) {
      CFGNode* D = stack.top(); stack.pop();
      if (body.find(D) == body.end()) {
        body.insert(D);
        for (CFGNode* pred : D->preds) {
          stack.push(pred);
        }
      }
    }
  }

  return true;
}

// Detects if-then-else branching patterns and records the root node (the one
// containing an if-then-else) and the join node (the block for code after
// the if-then-else)
//
// Basically finds places where a node has a conditional branch to two nodes
// that it dominates. Thus, they will not dominate the join point, but the
// join will post-dominate them and will be dominated by the root
static bool computeIfThenElse(std::vector<CFGNode*>& cfg,
    std::vector<std::pair<std::set<CFGNode*>, std::set<CFGNode*>>>& dominances,
    std::map<CFGNode*, std::pair<CFGNode*, std::set<CFGNode*>>>& loops,
    std::map<CFGNode*, CFGNode*>& result) {
  for (CFGNode* node : cfg) {
    if (node->conditional) {
      CFGNode* succ0 = node->succs[0];
      if (succ0 == nullptr) continue;
      std::set<CFGNode*>& domsSucc0 = dominances[succ0->index].first;
      CFGNode* succ1 = node->succs[1];
      if (succ1 == nullptr) continue;
      std::set<CFGNode*>& domsSucc1 = dominances[succ1->index].first;
  
      // If the node properly dominates both of its successors
      if (node != succ0 && domsSucc0.find(node) != domsSucc0.end()
       && node != succ1 && domsSucc1.find(node) != domsSucc1.end()) {
        std::set<CFGNode*>& postDomsSucc0 = dominances[succ0->index].second;
        std::set<CFGNode*>& postDomsSucc1 = dominances[succ1->index].second;

        std::set<CFGNode*> joined;
        std::set_intersection(postDomsSucc0.begin(), postDomsSucc0.end(),
                              postDomsSucc1.begin(), postDomsSucc1.end(),
                              std::inserter(joined, joined.begin()));
        
        // Now, joined contains the CFGNodes that post-dominate both paths, we
        // now go through all of them to find the first
        CFGNode* join = nullptr;
        for (CFGNode* cand : joined) {
          if (!join) {
            join = cand;
            continue;
          }

          std::set<CFGNode*>& postDomsJoin = dominances[join->index].second;
          std::set<CFGNode*>& postDomsCand = dominances[cand->index].second;

          if (postDomsJoin.find(cand) != postDomsJoin.end()) {
            // cand post-dominates current join point, ignore cand
            continue;
          } else if (postDomsCand.find(join) != postDomsCand.end()) {
            // current join point post-dominates cand, use cand as new join
            join = node;
          } else {
            std::cerr << "In computing if-then-else, two potential join "
                         "points did not have a post-dominance relation ("
                         __FILE__ " : " << __LINE__ << ")\n";
            return false;
          }
        }

        if (join == nullptr) {
          std::cerr << "Did not find join-point of an if-then-else (" __FILE__
                       " : " << __LINE__ << ")\n";
          return false;
        } else if (join == node) {
          std::cerr << "Join-point of if-then-else is the root node (" __FILE__
                       " : " << __LINE__ << ")\n";
          return false;
        }
        result[node] = join;
      }
    }
  }

  return true;
}

static bool emitControlFlow(CFGNode* start,
    std::vector<std::pair<std::set<CFGNode*>, std::set<CFGNode*>>>& dominances,
    std::map<CFGNode*, std::pair<CFGNode*, std::set<CFGNode*>>>& loops,
    std::map<CFGNode*, CFGNode*>& ifThenElse,
    std::string& output, std::vector<std::string>& blockCode,
    std::vector<InferType>& types, CFGNode* curLoop=nullptr);

static bool emitIfThenElse(CFGNode* split, std::string condName,
    std::vector<std::pair<std::set<CFGNode*>, std::set<CFGNode*>>>& dominances,
    std::map<CFGNode*, std::pair<CFGNode*, std::set<CFGNode*>>>& loops,
    std::map<CFGNode*, CFGNode*>& ifThenElse,
    std::string& output, std::vector<std::string>& blockCode,
    std::vector<InferType>& types, CFGNode* curLoop) {
  CFGNode* join = ifThenElse[split];

  output += "if (bool(" + condName + ")) {\n";
  if (!emitControlFlow(split->succs[0], dominances, loops, ifThenElse,
                       output, blockCode, types, curLoop)) return false;
  output += "} else {\n";
  if (!emitControlFlow(split->succs[1], dominances, loops, ifThenElse,
                       output, blockCode, types, curLoop)) return false;
  output += "}\n";

  return emitControlFlow(join, dominances, loops, ifThenElse, output,
                         blockCode, types, curLoop);
}

static bool emitNonIfThenElse(CFGNode* split, std::string condName,
    std::vector<std::pair<std::set<CFGNode*>, std::set<CFGNode*>>>& dominances,
    std::map<CFGNode*, std::pair<CFGNode*, std::set<CFGNode*>>>& loops,
    std::map<CFGNode*, CFGNode*>& ifThenElse,
    std::string& output, std::vector<std::string>& blockCode,
    std::vector<InferType>& types, CFGNode* curLoop) {
  output += "if (bool(" + condName + ")) {\n";

  CFGNode* loopExit = curLoop ? loops[curLoop].first : nullptr;

  CFGNode* succ0 = split->succs[0];
  if (succ0 == nullptr) output += "return;\n"; 
  else {
    std::set<CFGNode*>& domsSucc0 = dominances[succ0->index].first;
    if (succ0 == curLoop) {
      output += "continue;\n";
    } else if (succ0 == loopExit) {
      output += "break;\n";
    } else if (domsSucc0.find(split) != domsSucc0.end()) {
      // If the split dominates the node beneath it, code-gen that node into
      // the conditional
      if (!emitControlFlow(succ0, dominances, loops, ifThenElse, output,
                           blockCode, types, curLoop)) return false;
    } else {
      // Must be falling out of a containing if-then-else
    }
  }

  output += "} else {\n";

  CFGNode* succ1 = split->succs[1];
  if (succ1 == nullptr) output += "return;\n"; 
  else {
    std::set<CFGNode*>& domsSucc1 = dominances[succ1->index].first;
    if (succ1 == curLoop) {
      output += "continue;\n";
    } else if (succ1 == loopExit) {
      output += "break;\n";
    } else if (domsSucc1.find(split) != domsSucc1.end()) {
      if (!emitControlFlow(succ1, dominances, loops, ifThenElse, output,
                           blockCode, types, curLoop)) return false;
    } else {
      // Must be falling out of a containing if-then-else
    }
  }

  output += "}\n";

  return true;
}

static bool emitControlFlow(CFGNode* start,
    std::vector<std::pair<std::set<CFGNode*>, std::set<CFGNode*>>>& dominances,
    std::map<CFGNode*, std::pair<CFGNode*, std::set<CFGNode*>>>& loops,
    std::map<CFGNode*, CFGNode*>& ifThenElse,
    std::string& output, std::vector<std::string>& blockCode,
    std::vector<InferType>& types, CFGNode* curLoop) {
  
  auto fLoop = loops.find(start);
  if (fLoop != loops.end()) {
    CFGNode* exit = fLoop->second.first;
    output += "loop {\n";
    output += blockCode[start->index];

    if (!start->conditional) {
      if (start->succs[0] == start) {
        std::cerr << "Infinite loop detected (" __FILE__ " : " << __LINE__
                  << ")\n";
        return false;
      } else if (start->succs[0] == exit) {
        std::cerr << "Loop with only one iteration detected (WARNING: "
                     __FILE__ " : " << __LINE__ << ")\n";
        output += "break;\n";
      } else {
        if (start->succs[0] == nullptr) output += "return;\n";
        else if (!emitControlFlow(start->succs[0], dominances, loops,
                                  ifThenElse, output, blockCode, types, start))
                                    return false;
      }
    } else {
      std::string condName; InferType condType;
      if (!emitCode(start->cond, output, types, &condName, &condType))
        return false;
      if (condType == InferType::noType || condName == "") {
        std::cerr << "Condition did not produce a value (" __FILE__ " : "
                  << __LINE__ << ")\n";
        return false;
      }

      auto fIf = ifThenElse.find(start);
      if (fIf != ifThenElse.end()) {
        if (!emitIfThenElse(start, condName, dominances, loops, ifThenElse,
                            output, blockCode, types, start)) return false;
      } else {
        if (!emitNonIfThenElse(start, condName, dominances, loops, ifThenElse,
                               output, blockCode, types, start)) return false;
      }
    }

    output += "}\n";
    return emitControlFlow(exit, dominances, loops, ifThenElse, output,
                           blockCode, types, start);
  } else {
    output += blockCode[start->index];

    if (!start->conditional) {
      CFGNode* succ = start->succs[0];
      if (succ == nullptr) output += "return;\n";
      else {
        std::set<CFGNode*>& succDoms = dominances[succ->index].first;
        if (succ == curLoop) output += "continue;\n";
        else if (curLoop && succ == loops[curLoop].first) output += "break;\n";
        else if (succDoms.find(start) != succDoms.end()) {
          return emitControlFlow(succ, dominances, loops, ifThenElse, output,
                                 blockCode, types, curLoop);
        } else {
          // Ignore, must be exiting an if-then-else
        }
      }
    } else {
      std::string condName; InferType condType;
      if (!emitCode(start->cond, output, types, &condName, &condType))
        return false;
      if (condType == InferType::noType || condName == "") {
        std::cerr << "Condition did not produce a value (" __FILE__ " : "
                  << __LINE__ << ")\n";
        return false;
      }
      
      auto fIf = ifThenElse.find(start);
      if (fIf != ifThenElse.end()) {
        return emitIfThenElse(start, condName, dominances, loops, ifThenElse,
                              output, blockCode, types, curLoop);
      } else {
        return emitNonIfThenElse(start, condName, dominances, loops,
                                 ifThenElse, output, blockCode, types,
                                 curLoop);
      }
    }
  }

  return true;
}

static bool codeGen(Expression* expr, std::vector<InferType>& types,
                    std::string& output) {
  std::vector<CFGNode*> cfg;
  if (!constructCFG(expr, cfg)) return false;

  CFGNode* startNode = cfg[0];
  CFGNode* endNode = nullptr;

  std::vector<std::string> blockCode;
  for (CFGNode* node : cfg) {
    std::string block;
    if (!emitCode(*node, block, types)) return false;
    blockCode.push_back(block);
    if (node->succs[0] == nullptr) {
      if (endNode) {
        std::cerr << "Found multiple end nodes\n";
        return false;
      }
      endNode = node;
    }
  }

  if (!endNode) {
    std::cerr << "Found not end node\n";
    return false;
  }

  std::vector<std::pair<std::set<CFGNode*>, std::set<CFGNode*>>>
    dominances(cfg.size(),
               std::make_pair(std::set<CFGNode*>(), std::set<CFGNode*>()));
  std::vector<std::pair<CFGNode*, CFGNode*>> backEdges;
  std::map<CFGNode*, std::pair<CFGNode*, std::set<CFGNode*>>> loops;
  std::map<CFGNode*, CFGNode*> ifThenElse;

  computeDominances(cfg, startNode, endNode, dominances);
  if (!computeBackEdges(cfg, dominances, backEdges)) return false;
  if (!computeLoops(backEdges, loops)) return false;
  if (!computeIfThenElse(cfg, dominances, loops, ifThenElse)) return false;

  if (!emitControlFlow(startNode, dominances, loops, ifThenElse, output,
                       blockCode, types)) return false;

  for (CFGNode* node : cfg) delete node;
  return true;
}

static bool translateFunction(
      Function* func, std::vector<PassedType> const& types,
      const std::string& workgroup, std::string& result) {
  if (!func->type.isSignature()) return false;

  const Type& params = func->type.getSignature().params;

  std::vector<InferType> variableTypes;
  const int numArgs = params.size();
  const int numLocals = func->vars.size();

  if ((unsigned long) numArgs != types.size()) {
    std::cerr << "Number of function arguments (" << numArgs
              << ") and number of provided types (" << types.size()
              << ") differ, not supported (" << __FILE__ << " : " << __LINE__
              << ")\n";
    return false;
  }

  int index[3] = {-1, -1, -1};
  int dimension[3] = {-1, -1, -1};

  int i = 0;
  for (auto it = params.begin(), end = params.end();
       it != end; ++it, ++i) {
    const Type& param = *it;
    if (!param.isBasic()) {
      std::cerr << "Function argument is not a basic type, not supported ("
                << __FILE__ << " : " << __LINE__ << ")\n";
      return false;
    }

    const Type::BasicType bTy = param.getBasic();
    switch (bTy) {
      CASE_TYPE(i32, PROVIDED(i32) PROVIDED(u32) PROVIDED(i32_ptr)
                     PROVIDED_PTR(u32) PROVIDED_PTR(i64) PROVIDED_PTR(u64)
                     PROVIDED_PTR(f32) PROVIDED_PTR(f64)
                     PROVIDED_IDX(x) PROVIDED_IDX(y) PROVIDED_IDX(z)
                     PROVIDED_DIM(x) PROVIDED_DIM(y) PROVIDED_DIM(z))
      CASE_TYPE(i64, PROVIDED(i64) PROVIDED(u64))
      CASE_TYPE(f32, PROVIDED(f32))
      CASE_TYPE(f64, PROVIDED(f64))
      default:
        std::cerr << "Function argument is of type " << param.toString()
                  << ", not supported (" << __FILE__ << " : " << __LINE__
                  << ")\n";
        return false;
    }
  }

  if (!locateIdxDim(variableTypes, index, dimension)) return false;

  for (const Type& local : func->vars) {
    if (!local.isBasic()) {
      std::cerr << "Local variable is not a basic type, not supported ("
                << __FILE__ << " : " << __LINE__ << ")\n";
      return false;
    }

    const Type::BasicType bTy = local.getBasic();
    switch (bTy) {
      // For some reason, Emscripten doesn't seem to generate unsinged types,
      // so we mark i32 and i64 as possibly signed or unsigned, we'll
      // differentiate based on interations with known sign values
      CASE_TYPE(i32, ASSIGN(n32))
      CASE_TYPE(i64, ASSIGN(n64))
      CASE_TYPE(f32, ASSIGN(f32))
      CASE_TYPE(f64, ASSIGN(f64))
      default:
        std::cerr << "Local variable is of type " << local.toString()
                  << ", not supported (" << __FILE__ << " : " << __LINE__
                  << ")\n";
        return false;
    }
  }

  std::string body;
  if (codeGen(func->body, variableTypes, body)) {
    std::string typeDecls, bindDecls, signature, argInits, footer;
    std::string scalarFields;
    int numBinds = 0;
    for (int i = 0; i < numArgs; i++) {
      InferType ty = variableTypes[i];
      std::string typeDecl, argDecl, argInit;
      if (!types[i].idx) {
        if (isArray(ty)) {
          typeDecl = "struct Arg" + std::to_string(i) + " { value : "
                   + typeToWebGPU(ty) + " };\n";
          argDecl = "@group(0) @binding(" + std::to_string(numBinds)
                  + ") var<storage, "
                  + (types[i].write ? "read_write" : "read")
                  + "> arg" + std::to_string(i) + " : Arg" + std::to_string(i)
                  + ";\n";
          // At least currently there are some issues with putting an array
          // into a local, so we'll have to just use these by name
          // like argN.value
          //argInit = "\tvar local" + std::to_string(i) + " : "
          //        + typeToWebGPU(ty) + " = arg" + std::to_string(i)
          //        + ".value;\n";
          numBinds++;
        } else {
          // There's a limit on the number of bindings allowed (I don't know if
          // this is a Vulkan limitation, a WebGPU limitation, or an
          // implementation limitation, but to deal with this we put all the
          // scalars into a single struct
          scalarFields += "\targ" + std::to_string(i) + " : "
                        + typeToWebGPU(ty) + ",\n";
          argInit = "\tvar local" + std::to_string(i) + " : "
                  + typeToWebGPU(ty) + " = scalars.arg" + std::to_string(i)
                  + ";\n";
        }
      } else {
        argInit = "\tvar local" + std::to_string(i) + " : "
                + typeToWebGPU(ty) + " = global_id." + types[i].type + ";\n";
      }

      typeDecls += typeDecl;
      bindDecls += argDecl;
      argInits += argInit;
    }
    for (int i = numArgs; i < numArgs + numLocals; i++) {
      argInits += "\tvar local" + std::to_string(i) + " : "
                + typeToWebGPU(variableTypes[i]) + " = "
                + typeToWebGPU(variableTypes[i]) + "(0);\n";
    }

    typeDecls += "struct Scalars {\n" + scalarFields + "};\n";
    bindDecls += "@group(0) @binding(" + std::to_string(numBinds)
               + ") var<storage, read> scalars : Scalars;\n";

    signature =
        "@stage(compute) @workgroup_size" + workgroup + "\n"
        "fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {\n";
    footer = "}";
    
    std::string indexGuard = "if (local" + std::to_string(index[0]) + " >= "
                             "local" + std::to_string(dimension[0]) + " || "
                             "local" + std::to_string(index[1]) + " >= "
                             "local" + std::to_string(dimension[1]) + " || "
                             "local" + std::to_string(index[2]) + " >= "
                             "local" + std::to_string(dimension[2]) + ") {\n"
                             "\treturn;\n"
                             "}\n";
    
    result = typeDecls + "\n" + bindDecls + "\n" + signature + argInits
           + indexGuard + body + footer;
    return true;
  }

  return false;
}

struct HPVMWebGPU : public Pass {
  void run(PassRunner* runner, Module* module) override {
    auto& options = runner->options;

    std::string kernelName =
      options.getArgumentOrDefault("hpvm.kernel.name", "");
    std::string kernelType =
      options.getArgumentOrDefault("hpvm.kernel.types", "");
    std::string workgroupSize =
      options.getArgumentOrDefault("hpvm.kernel.workgroup", "");

    if (kernelName == "") {
      std::cerr << "No kernel function name given\n";
      options.arguments["hpvm.kernel.error"] = "true";
      return;
    } else if (kernelType == "") {
      std::cerr << "Kernel type not provided\n";
      options.arguments["hpvm.kernel.error"] = "true";
      return;
    } else if (workgroupSize == "") {
      std::cerr << "Kernel workgroup size not provided\n";
      options.arguments["hpvm.kernel.error"] = "true";
      return;
    }

    Name functionName;
    if (!findKernel(module, kernelName, functionName)) {
      std::cerr << "Kernel not found in module\n";
      options.arguments["hpvm.kernel.error"] = "true";
      return;
    }

    std::vector<PassedType> types;
    if (!extractTypes(kernelType, types)) {
      options.arguments["hpvm.kernel.error"] = "true";
      return;
    }

    Function* kernelFunc = module->getFunction(functionName);
    std::string result;
    if (translateFunction(kernelFunc, types, workgroupSize, result)) {
      options.arguments["hpvm.kernel.error"] = "false";
      options.arguments["hpvm.kernel.result"] = result;
    } else {
      options.arguments["hpvm.kernel.error"] = "true";
    }
  }
};

Pass* createHPVMWebGPUPass() { return new HPVMWebGPU(); }

} // namespace wasm
