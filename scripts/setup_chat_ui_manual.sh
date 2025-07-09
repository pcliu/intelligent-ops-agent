#!/bin/bash

# Agent Chat UI 手动设置脚本（更简单的方法）

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    echo -e "${2}${1}${NC}"
}

print_message "🎨 Agent Chat UI 手动设置脚本" "$BLUE"
echo

# 检查 Node.js 和 pnpm
if ! command -v node &> /dev/null; then
    print_message "❌ Node.js 未安装，请先安装 Node.js" "$RED"
    exit 1
fi

if ! command -v pnpm &> /dev/null; then
    print_message "❌ pnpm 未安装，正在安装..." "$YELLOW"
    npm install -g pnpm
    print_message "✅ pnpm 安装完成" "$GREEN"
fi

# 解析命令行参数
PROJECT_NAME="intelligent-ops-chat-ui"
LANGGRAPH_URL="http://localhost:2024"
ASSISTANT_ID="ops-agent"

while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --langgraph-url)
            LANGGRAPH_URL="$2"
            shift 2
            ;;
        --assistant-id)
            ASSISTANT_ID="$2"
            shift 2
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo "选项:"
            echo "  --name NAME                设置项目名称 (默认: intelligent-ops-chat-ui)"
            echo "  --langgraph-url URL        设置 LangGraph 服务器地址 (默认: http://localhost:2024)"
            echo "  --assistant-id ID          设置助手 ID (默认: ops-agent)"
            echo "  --help                     显示此帮助信息"
            exit 0
            ;;
        *)
            print_message "❌ 未知选项: $1" "$RED"
            exit 1
            ;;
    esac
done

# 克隆仓库
print_message "📦 克隆 Agent Chat UI 仓库..." "$BLUE"
git clone https://github.com/langchain-ai/agent-chat-ui.git $PROJECT_NAME
cd $PROJECT_NAME

print_message "📦 安装依赖..." "$BLUE"
pnpm install

# 创建 .env.local 配置文件
print_message "🔧 创建配置文件..." "$BLUE"
cat > .env.local << EOF
# Next.js 应用配置
NEXT_PUBLIC_API_URL=http://localhost:3000/api
NEXT_PUBLIC_ASSISTANT_ID=$ASSISTANT_ID

# LangGraph 服务器配置
LANGGRAPH_API_URL=$LANGGRAPH_URL
LANGGRAPH_API_KEY=

# 可选配置
# LANGSMITH_API_KEY=your-langsmith-api-key
# LANGSMITH_PROJECT_NAME=intelligent-ops-chat

# 开发配置
NODE_ENV=development
EOF

print_message "✅ 配置文件 .env.local 创建完成" "$GREEN"

# 创建启动脚本
print_message "🚀 创建启动脚本..." "$BLUE"
cat > start_chat_ui.sh << 'EOF'
#!/bin/bash

# Chat UI 启动脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🎨 启动 Agent Chat UI...${NC}"
echo

# 检查 LangGraph 服务器
echo -e "${BLUE}📡 检查 LangGraph 服务器连接...${NC}"
if curl -s http://localhost:2024/health > /dev/null; then
    echo -e "${GREEN}✅ LangGraph 服务器正在运行${NC}"
else
    echo -e "${YELLOW}⚠️  LangGraph 服务器未运行，请先启动服务器${NC}"
    echo -e "${YELLOW}   运行命令: cd .. && ./scripts/start_server.sh${NC}"
    echo
fi

# 显示访问地址
echo -e "${BLUE}📍 Chat UI 地址: http://localhost:3000${NC}"
echo -e "${BLUE}📍 LangGraph API: http://localhost:2024${NC}"
echo

# 启动开发服务器
echo -e "${BLUE}🚀 启动开发服务器...${NC}"
pnpm dev
EOF

chmod +x start_chat_ui.sh

print_message "✅ 启动脚本创建完成" "$GREEN"

# 显示设置完成信息
echo
print_message "🎉 Agent Chat UI 设置完成!" "$GREEN"
echo
print_message "📋 下一步操作:" "$BLUE"
echo "1. 启动 LangGraph 服务器:"
echo "   cd .. && ./scripts/start_server.sh"
echo
echo "2. 启动 Chat UI:"
echo "   cd $PROJECT_NAME && ./start_chat_ui.sh"
echo "   或者: cd $PROJECT_NAME && pnpm dev"
echo
echo "3. 访问 Chat UI:"
echo "   http://localhost:3000"
echo

print_message "🔧 配置文件位置:" "$BLUE"
echo "- .env.local: Chat UI 配置"
echo "- start_chat_ui.sh: 启动脚本"
echo

print_message "💡 提示:" "$YELLOW"
echo "- 确保 LangGraph 服务器在 $LANGGRAPH_URL 运行"
echo "- 助手 ID 设置为: $ASSISTANT_ID"
echo "- 如需修改配置，请编辑 .env.local 文件"