---
title: 为你的hexo添加不止一个评论区
date: 2025-07-15 22:03:04
tags: hexo美化
description: 使用giscus和更新容器为你的hexo页面添加不止一个评论区
hidden: false
---
## 一、hexo添加评论区
- hexo添加评论区还比较容易地,一共步

&emsp;&emsp;我使用的是giscus,原理是使用github库中的discussions部分映射到你的博客中,你在博客的评论会储存到被映射库的discussions中.
&emsp;&emsp;相同原理的也有utterances,它是使用github库中的issue部分映射的,但是utterances相对来说使用较为麻烦
### 1、为你的想要映射的库添加discussions
&emsp;&emsp;你可以使用你博客的github.io,也可以新建一个库用于储存评论(如果你愿意的话)
点击**这个库**的setting
![](/posts/为你的hexo添加不止一个评论区/添加discussion.png)
在general中直接往下滑,找到discussion,点击新建,就成功的为你这个库添加了discussion
![](/posts/为你的hexo添加不止一个评论区/image.png)
### 2、在你的库中下载giscus
&emsp;&emsp;点击[giscus GitHub App](https://github.com/apps/giscus)进入,点击下载就ok,显示如下:
![](/posts/为你的hexo添加不止一个评论区/image3.png)
&emsp;&emsp;如果之后再想加入新的仓库下载giscus进入就再个人-设置-Applications-configure加入新的仓库即可
### 3、启用 giscus
&emsp;&emsp;点击[giscus.app](https://giscus.app/zh-CN)进入,
![](/posts/为你的hexo添加不止一个评论区/image2.png)
&emsp;&emsp;输入你的用户名和仓库名,如果正确会显示
&emsp;&emsp;(√,成功！该仓库满足所有条件。)
&emsp;&emsp;下面的是映射反应,pathname和URL和title都相差不大,基于每个页面提供一个独特的标题
&emsp;&emsp;如果你想要在一个页面里加入多个评论区,需要使用"Discussion 的标题包含特定字符串"
&emsp;&emsp;Discussion 分类选择Announcements,最后参考下面的Js修改你的config.butterfly的comment部分
```yml
giscus:
  enable: true
  repo: "Morgenro/Morgenro.github.io"
  repo_id: "Your_repo_id"
  category: "Announcements"
  category_id: "Your_category_id"
  light_theme: light
  dark_theme: dark
  js:
  option:
  ```
```上述的是博客默认的评论区,如果你想要在其他页面添加comments,在md中加入js就好```
```js
 <div class="giscus-wrapper">
      <script src="https://giscus.app/client.js"
        data-repo="Morgenro/Morgenro.github.io"
        data-repo-id="Your_data-repo-id"
        data-category="Announcements"
        data-category-id="Your_category-id"
        data-mapping="specific" 
        data-term="diary-2023-07-14" 
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="bottom"
        data-theme="light"
        data-lang="zh-CN"
        crossorigin="anonymous"
        async>
      </script>
```
到这里就实现了hexo添加评论区
## 二、为你的博客页面添加多个评论区
&emsp;&emsp;当我直接在一个页面加入两个基于字符串的评论区时,发现只会显示最下面的评论区,原因是
&emsp;&emsp;Giscus 插件（尤其是 Hexo 中的 giscus 插件）生成的是静态 HTML，只能在构建阶段加载一次脚本；
&emsp;&emsp;多个实例无法动态挂载，因为 script 只执行一次，不支持多个容器同时绑定多个 Giscus 实例。
&emsp;&emsp;这时候想到使用iframe,在一个网页中嵌入多个 iframe,分别加载不同的页面并在这些页面中各自加载 Giscus。尝试之后发现确实出现了两个评论区的轮廓,但是其中显示giscus.app未连接,原因可能是不支持跨域通信,也罢
&emsp;&emsp;于是就干脆只显示一个评论区,在查看下一个评论区时将上面的评论区销毁即可,代码如下
```js
  let currentGiscusContainer = null;

  function loadGiscus(targetElement, term) {
    // 清空之前的评论容器
    if (currentGiscusContainer) {
      currentGiscusContainer.innerHTML = '';
    }

    // 如果不存在容器则创建一个
    let giscusDiv = targetElement.querySelector('.giscus-container');
    if (!giscusDiv) {
      giscusDiv = document.createElement('div');
      giscusDiv.className = 'giscus-container';
      targetElement.appendChild(giscusDiv);
    }

    currentGiscusContainer = giscusDiv;
    giscusDiv.innerHTML = ''; // 清空旧的内容

    const script = document.createElement('script');
    script.src = 'https://giscus.app/client.js';
    script.setAttribute('data-repo', giscusConfig.repo);
    script.setAttribute('data-repo-id', giscusConfig.repoId);
    script.setAttribute('data-category', giscusConfig.category);
    script.setAttribute('data-category-id', giscusConfig.categoryId);
    script.setAttribute('data-mapping', giscusConfig.mapping);
    script.setAttribute('data-term', term);
    script.setAttribute('data-strict', giscusConfig.strict);
    script.setAttribute('data-reactions-enabled', giscusConfig.reactionsEnabled);
    script.setAttribute('data-emit-metadata', giscusConfig.emitMetadata);
    script.setAttribute('data-input-position', giscusConfig.inputPosition);
    script.setAttribute('data-theme', giscusConfig.theme);
    script.setAttribute('data-lang', giscusConfig.lang);
    script.crossOrigin = 'anonymous';
    script.async = true;

    giscusDiv.appendChild(script);
  }

  document.addEventListener('DOMContentLoaded', function () {
    const diaryItems = document.querySelectorAll('.diary-item');
    if (diaryItems.length > 0) {
      loadGiscus(diaryItems[0], diaryItems[0].getAttribute('data-term'));
    }

    diaryItems.forEach(item => {
      item.addEventListener('click', () => {
        const term = item.getAttribute('data-term');
        loadGiscus(item, term);
        item.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    });
  });
```
于此就实现了在你需要的网页添加多个评论区
```这里是点击触发,你也可以添加<bottom>来实现"展开评论"的按钮```
## 7-18更新
&emsp;&emsp;发现日记的评论区有时不会刷新在卡片中，而是刷新在网页的最下方，这是因为如果你在md文档中设置了comments= true，我使用的是loadGiscus函数动态加载，评论区脚本就会被被重复加载。导致评论区刷新位置出现问题。