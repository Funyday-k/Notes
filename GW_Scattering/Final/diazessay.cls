\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{diazessay}[2019/01/13 Diaz Essay class v1.0]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}} % Pass through any options to the base class
\ProcessOptions\relax % Process given options

\LoadClass{article} % Load the base class

%----------------------------------------------------------------------------------------
%	PACKAGES AND OTHER DOCUMENT CONFIGURATIONS
%----------------------------------------------------------------------------------------

\usepackage[english]{babel}




\usepackage{amsmath,amssymb,amsthm}
\usepackage{bookmark}
\usepackage{graphicx} % Required for including images
\graphicspath{{Figures/}{./}} % Specifies where to look for included images (trailing slash required)

\usepackage{wrapfig} % Allows in-line images

\usepackage{booktabs}

\usepackage{listings} 
\usepackage{xcolor}
\lstset{
  language=Matlab,  %代码语言使用的是matlab
  frame=shadowbox, %把代码用带有阴影的框圈起来
  rulesepcolor=\color{red!20!green!20!blue!20},%代码块边框为淡青色
  keywordstyle=\color{blue!90}\bfseries, %代码关键字的颜色为蓝色，粗体
  commentstyle=\color{red!10!green!70}\textit,    % 设置代码注释的颜色
  showstringspaces=false,%不显示代码字符串中间的空格标记
  numbers=left, % 显示行号
  numberstyle=\tiny,    % 行号字体
  stringstyle=\ttfamily, % 代码字符串的特殊格式
  breaklines=true, %对过长的代码自动换行
  extendedchars=false,  %解决代码跨页时，章节标题，页眉等汉字不显示的问题
%   escapebegin=\begin{CJK*},escapeend=\end{CJK*},      % 代码中出现中文必须加上，否则报错
  texcl=true}


\usepackage{enumitem} % Required for list customisation
\setlist{noitemsep} % No spacing between list items


\makeatletter
\renewcommand\@biblabel[1]{\textbf{#1.}} % Remove the square brackets from each bibliography item ('[1]' to '1.')
\makeatother

%----------------------------------------------------------------------------------------
%	DOCUMENT MARGINS
%----------------------------------------------------------------------------------------

\usepackage{geometry} % Required for adjusting page dimensions and margins

\geometry{
	paper=a4paper, % Paper size, change to letterpaper for US letter size
	top=2.5cm, % Top margin
	bottom=3cm, % Bottom margin
	left=2.5cm, % Left margin
	right=2.5cm, % Right margin
	headheight=0.75cm, % Header height
	footskip=1.5cm, % Space from the bottom margin to the baseline of the footer
	headsep=0.75cm, % Space from the top margin to the baseline of the header
	%showframe, % Uncomment to show how the type block is set on the page
}

%----------------------------------------------------------------------------------------
%	FONTS
%----------------------------------------------------------------------------------------

\usepackage[utf8]{inputenc} % Required for inputting international characters
\usepackage[T1]{fontenc} % Use 8-bit encoding

\usepackage{mathpazo} % Use the Palatino font


\linespread{1.05} % Increase line spacing slightly; Palatino benefits from a slight increase by default

%----------------------------------------------------------------------------------------
%	TITLE
%----------------------------------------------------------------------------------------

\makeatletter
\renewcommand{\maketitle}{ % Customise the title (do not edit title and author name here)
	\vspace*{30pt} % Vertical whitespace before the title
	\begin{flushleft} % Right align
		{\LARGE\@title} % Increase the font size of the title
		
		\vspace{50pt} % Vertical whitespace between the title and author name
		
		{\large\@author}\\ % Author name
		\@date % Date
		
		\vspace{40pt} % Vertical whitespace between the author block and abstract
	\end{flushleft}
}
\makeatother

%----------------------------------------------------------------------------------------
%	HEADERS AND FOOTERS
%----------------------------------------------------------------------------------------

\usepackage{fancyhdr} % Required for customising headers and footers
\pagestyle{fancy} % Enable custom headers and footers

\renewcommand{\headrulewidth}{1pt} % Remove default top horizontal rule
\lhead{\textbf{Lingyu Xia}} % Left header
\chead{} % Centre header
\rhead{\textbf{GW Scattering Calculation Note}} % Right header
\lfoot{} % Left footer
\cfoot{\thepage} % Centre footer
\rfoot{} % Right footer


\usepackage{hyperref}
\hypersetup{colorlinks = true, linktoc=all, linkcolor=black, urlcolor=winered}
